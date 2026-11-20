(() => {
  const actions = ["up", "right", "down", "left"];
  const offsets = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  const keyCodes = [38, 39, 40, 37];
  const ghostIds = ["blinky", "pinky", "inky", "clyde"];
  const POWER = 1;
  const FRUIT = 2;
  const EDIBLE_GHOST = 3;
  const byId = (id) => document.getElementById(id);

  const panel = document.createElement("div");
  panel.id = "pacman-rl-demo";
  const shadow = panel.attachShadow({ mode: "open" });
  shadow.innerHTML = `
    <style>
      :host { position: fixed; z-index: 2147483647; top: 12px; right: 12px; }
      section { background: #17191f; color: #f5f5f5; padding: 10px;
                border: 1px solid #696c75; border-radius: 8px; font: 14px system-ui; }
      button { padding: 5px 9px; font: inherit; cursor: pointer; }
      output { display: block; max-width: 210px; margin-top: 5px; }
    </style>
    <section aria-label="Pac-Man reinforcement learning demo">
      <button type="button">Start RL autoplay</button>
      <output aria-live="polite"></output>
    </section>
  `;
  document.body.append(panel);
  const button = shadow.querySelector("button");
  const status = shadow.querySelector("output");
  let frame = null;
  let session = 0;
  let active = false;
  let lastGhostEaten = -Infinity;

  window.addEventListener("eatGhost", () => {
    lastGhostEaten = performance.now();
  });

  function stop(message) {
    session++;
    active = false;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    button.textContent = "Start RL autoplay";
    status.value = message;
  }

  async function read(url, format) {
    const response = await fetch(url);
    if (!response.ok) throw new Error("Could not load " + format);
    return format === "policy" ? response.json() : response.text();
  }

  async function loadPolicy() {
    const policy = await read(chrome.runtime.getURL("policy.json"), "policy");
    if (policy.version !== 3 || policy.actions?.join() !== actions.join()
        || !policy.q || Object.keys(policy.q).length === 0
        || Object.entries(policy.q).some(([state, values]) =>
          !/^\d+:\d+:\d+:(-1|[0-3]):[0-3]:(-1|[0-3])$/.test(state) || !Array.isArray(values)
          || values.length !== 4 || !values.every(Number.isFinite))) {
      throw new Error("Run the notebook to export the policy, then reload the extension");
    }
    return policy.q;
  }

  async function loadGraph() {
    // shortcut: this parser depends on the site's current bundle format, update it if the site changes.
    const bundle = await read(new URL("/build/app.js", location.origin), "maze");
    const literal = bundle.match(/this\.mazeArray\s*=\s*\[([\s\S]*?)\]\s*;/)?.[1];
    const rows = Array.from((literal || "").matchAll(/\[\s*['"]([A-Za-z .\-]+)['"]\s*\]/g),
      (match) => match[1]);
    const width = 28;
    const height = 31;
    if (rows.length !== height || rows.some((row) => row.length !== width)) {
      throw new Error("FreePacman maze format changed");
    }
    const graph = new Map();
    rows.forEach((row, y) => {
      Array.from(row).forEach((cell, x) => {
        if (cell === "X") return;
        const edges = [];
        offsets.forEach(([dx, dy], action) => {
          const nextX = (x + dx + width) % width;
          const nextY = y + dy;
          if (rows[nextY] && rows[nextY][nextX] !== "X") {
            edges.push([action, nextY * width + nextX]);
          }
        });
        graph.set(y * width + x, edges);
      });
    });
    return { graph, width, height };
  }

  function routesFrom(graph, start, limit = Infinity, firstMoves = graph.get(start)) {
    const routes = new Map([[start, [0, -1]]]);
    for (const [tile, [distance, first]] of routes) {
      if (distance >= limit) continue;
      for (const [action, neighbor] of distance ? graph.get(tile) : firstMoves) {
        if (!routes.has(neighbor)) routes.set(neighbor, [distance + 1, distance ? first : action]);
      }
    }
    return routes;
  }

  function firstTarget(routes, targets, limit = Infinity) {
    for (const [tile, [distance, first]] of routes) {
      if (distance > limit) break;
      if (distance && targets.has(tile)) return first;
    }
    return -1;
  }

  function corridorRisk(graph, player, next, threat, power) {
    const gap = threat.get(next)?.[0] ?? Infinity;
    if (gap <= 1 && (gap === 0 || !power.has(next))) return 2;
    const current = threat.get(player)?.[0] ?? Infinity;
    let risk = Number(gap <= 2 && gap < current && !power.has(next));
    let previous = player;
    let tile = next;
    for (let depth = 1; depth <= 3 && !power.has(tile); depth++) {
      const distance = threat.get(tile)?.[0] ?? Infinity;
      const onward = graph.get(tile).filter(([, target]) => target !== previous);
      if ((depth > 1 && distance <= depth) || (!onward.length && distance <= 2 * depth + 2)) {
        risk = 1;
      }
      if (onward.length !== 1) break;
      previous = tile;
      tile = onward[0][1];
    }
    return risk;
  }

  function decide(graph, player, ghosts, pellets, bonuses, table, seekPellet) {
    const moves = graph.get(player);
    if (!moves?.length) return null;
    const routes = routesFrom(graph, player);
    const nearest = firstTarget(routes, pellets);
    let bonus = EDIBLE_GHOST;
    let direction = firstTarget(routes, bonuses.edible, 8);
    if (direction < 0) {
      bonus = FRUIT;
      direction = firstTarget(routes, bonuses.fruit, 12);
    }
    if (direction < 0) {
      bonus = POWER;
      direction = firstTarget(routes, bonuses.power, 8);
    }
    if (direction < 0) bonus = 0;
    const threats = ghosts.map((ghost) => routesFrom(graph, ghost, 10))
      .filter((threat) => threat.has(player));
    let valid = 0;
    let danger = 0;
    let caution = 0;
    let lowestRisk = Infinity;
    let bestGap = -1;
    let choices = [];
    for (const move of moves) {
      const [action, next] = move;
      valid |= 1 << action;
      const risk = Math.max(0, ...threats.map((threat) =>
        corridorRisk(graph, player, next, threat, bonuses.power)));
      const gap = risk ? Math.min(...threats.map((threat) => threat.get(next)?.[0] ?? Infinity)) : 0;
      if (risk === 2) danger |= 1 << action;
      if (risk === 1) caution |= 1 << action;
      if (risk < lowestRisk || (risk === lowestRisk && gap > bestGap)) {
        lowestRisk = risk;
        bestGap = gap;
        choices = [];
      }
      if (risk === lowestRisk && gap === bestGap) choices.push(move);
    }
    const values = table[`${valid}:${danger}:${caution}:${nearest}:${bonus}:${direction}`];
    if (!values || seekPellet) {
      if (!seekPellet && choices.some(([action]) => action === direction)) return direction;
      const target = choices.length === moves.length ? nearest
        : firstTarget(routesFrom(graph, player, Infinity, choices), pellets);
      if (target >= 0) return target;
    }
    let best = choices[0][0];
    for (const [action] of choices) {
      if (values && values[action] > values[best]) best = action;
    }
    return best;
  }

  function visible(element) {
    return element && getComputedStyle(element).visibility === "visible";
  }

  function play(board, table) {
    const { graph, width, height } = board;
    const maze = byId("maze");
    const sprite = byId("pacman");
    const dotContainer = byId("dot-container");
    const pause = byId("pause-button");
    const menu = byId("main-menu-container");
    if (!maze || !sprite || !dotContainer?.childElementCount || !pause || !menu || visible(menu)) {
      stop("Press Play in FreePacman first.");
      return;
    }
    const ghosts = ghostIds.map(byId);
    if (ghosts.some((ghost) => !ghost)) {
      stop("Stopped. Could not find the ghosts.");
      return;
    }
    const dots = Array.from(dotContainer.children);
    const fruit = Array.from(maze.children).find((element) =>
      element.style.backgroundImage.includes("/pickups/"));
    let lastPlayer = null;
    let previousPlayer = null;
    let lastInput = 0;
    let lastPelletCount = null;
    let lastPelletAt = 0;
    let seekPellet = false;
    let unreadableSince = null;

    function tick(now) {
      if (!maze.isConnected || !sprite.isConnected || !dotContainer.isConnected || visible(menu)) {
        stop("Game ended. Press Play to try again.");
        return;
      }
      // The site hides Pac-Man and freezes movement for one second after a ghost is eaten.
      if (document.hidden || pause.textContent.trim() === "play_arrow" || now - lastGhostEaten < 1100) {
        lastPlayer = null;
        previousPlayer = null;
        lastPelletCount = null;
        seekPellet = false;
        unreadableSince = null;
        frame = requestAnimationFrame(tick);
        return;
      }
      const tileSize = parseFloat(maze.style.width) / width;
      const tileAt = (element, offset) => {
        const x = Math.round(parseFloat(element.style.left) / tileSize + offset);
        const y = Math.round(parseFloat(element.style.top) / tileSize + offset);
        if (!(x >= -1 && x <= width && y >= 0 && y < height)) return null;
        const tile = y * width + ((x + width) % width);
        return graph.has(tile) ? tile : null;
      };
      const player = visible(sprite) ? tileAt(sprite, 0.5) : null;
      if (player === null || !(tileSize > 0)) {
        unreadableSince ??= now;
        if (now - unreadableSince > 500) {
          stop("Stopped. Could not read Pac-Man's position.");
          return;
        }
      } else {
        unreadableSince = null;
        if (player !== lastPlayer || now - lastInput > 300) {
          const threats = [];
          const bonuses = { edible: new Set(), fruit: new Set(), power: new Set() };
          for (const ghost of ghosts.filter(visible)) {
            const tile = tileAt(ghost, 0.5);
            const image = ghost.style.backgroundImage;
            if (tile === null || image.includes("/ghosts/eyes_")) continue;
            if (image.includes("/ghosts/scared_")) bonuses.edible.add(tile);
            else threats.push(tile);
          }
          if (visible(fruit)) bonuses.fruit.add(tileAt(fruit, 0.5));
          const pellets = new Set();
          for (const dot of dots) {
            if (dot.style.visibility === "hidden") continue;
            const tile = tileAt(dot, 0);
            if (tile === null) continue;
            pellets.add(tile);
            if (dot.classList.contains("power-pellet")) bonuses.power.add(tile);
          }
          if (pellets.size !== lastPelletCount) {
            lastPelletCount = pellets.size;
            lastPelletAt = now;
            seekPellet = false;
          } else if ((player !== lastPlayer && player === previousPlayer) || now - lastPelletAt >= 1200) {
            seekPellet = true;
          }
          const action = decide(graph, player, threats, pellets, bonuses, table, seekPellet);
          if (action === null) {
            stop("Stopped. Could not find a valid move.");
            return;
          }
          const code = keyCodes[action];
          const key = "Arrow" + actions[action][0].toUpperCase() + actions[action].slice(1);
          window.dispatchEvent(new KeyboardEvent("keydown", { key, code: key, keyCode: code }));
          if (player !== lastPlayer) {
            previousPlayer = lastPlayer;
            lastPlayer = player;
          }
          lastInput = now;
        }
      }
      frame = requestAnimationFrame(tick);
    }

    status.value = "RL autoplay running.";
    frame = requestAnimationFrame(tick);
  }

  button.addEventListener("click", async () => {
    if (active) {
      stop("Autoplay stopped.");
      return;
    }
    const token = ++session;
    active = true;
    button.textContent = "Stop";
    status.value = "Loading trained policy…";
    try {
      const [board, table] = await Promise.all([loadGraph(), loadPolicy()]);
      if (session === token) play(board, table);
    } catch (error) {
      if (session === token) stop(error.message);
    }
  });
})();
