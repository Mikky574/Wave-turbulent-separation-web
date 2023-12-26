import json
from fastapi import FastAPI, HTTPException, WebSocket, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from main_function import main_function  # 导入修改过的 main 函数
from pathlib import Path

import uvicorn

app = FastAPI()
app.openapi_version = "3.0.3"

@app.post("/upload")
async def upload_file(
    u2: UploadFile, v2: UploadFile, w2: UploadFile, theta: UploadFile
) -> "dict[str, str]":
    file_paths: "dict[str,str]" = {}
    for file, desc in ((u2, "u2"), (v2, "v2"), (w2, "w2"), (theta, "theta")):
        contents = await file.read()
        upload_dir = Path(__file__).parent / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        filename = file.filename or "_"
        file_path = upload_dir / filename
        with open(file_path, "wb") as f:
            f.write(contents)
        file_paths[desc] = str(file_path)
    return file_paths


class FilePaths(BaseModel):
    u2: str
    v2: str
    w2: str
    theta: str


class WsData(BaseModel):
    action: str
    file_paths: FilePaths
    fs: float
    fw1: float
    fw2: float


@app.post("/__ws_placeholder")
def ws_placeholder(data: WsData) -> None:
    # 仅用于生成 OpenAPI 文档
    return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_json()  # 接收 JSON 数据

        if "action" in data and data["action"] == "start":
            data = WsData(**data)
            file_paths = data.file_paths
            Fs = data.fs
            Fw1 = data.fw1
            Fw2 = data.fw2

            # 调用 main 函数进行数据处理
            await main_function(
                websocket,
                file_paths.u2,
                file_paths.v2,
                file_paths.w2,
                file_paths.theta,
                Fs,
                Fw1,
                Fw2,
            )

            # 通知前端处理完成
            await websocket.send_text("Processing completed")
            break

    await websocket.close()


@app.get("/{file_path:path}", include_in_schema=False)
async def get_file(file_path: str):
    if not file_path:
        file_path = "index.html"
    path = Path(__file__).parent / "front" / "dist" / file_path
    if Path(__file__).parent not in path.parents or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


with (Path(__file__).parent / "openapi.json").open("w") as f:
    json.dump(app.openapi(), f, indent=1)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
