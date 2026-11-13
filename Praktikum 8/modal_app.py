import shlex
import subprocess
import modal

app = modal.App("fashion-mnist-gradio")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gradio",
        "numpy",
        "pillow",
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_file("app.py", "/root/app.py")
    .add_local_file("model.pth", "/root/model.pth")
)


@app.function(image=image, max_containers=1)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=60)
def web():
    cmd = f"GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=8000 python {shlex.quote('/root/app.py')}"
    subprocess.Popen(cmd, shell=True)