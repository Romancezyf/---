from flask import Flask, render_template, Response
import time
from utils import *
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

app = Flask(__name__)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)  # 初始化MTCNN模型进行人脸检测
# 初始化InceptionResnetV1模型进行人脸特征提取
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
dataset_path = r"C:\Users\Romance\Desktop\科研实践\dataset"
embeddings, labels = load_dataset(dataset_path, mtcnn, resnet)  # 提取给定图像的embedding，便于人脸检测的识别

cap = cv2.VideoCapture(0)  # 打开摄像头，这里不考虑采用多线程，直接使用公用的摄像头

def detect_faces():
    try:
        while True:
            ret, frame = cap.read()  # 读取帧
            if not ret:
                continue  # 如果读取失败，跳过
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 将帧转换为PIL图像
            boxes, probs = mtcnn.detect(img)
            if boxes is not None:
                for box in boxes:
                    # 提取人脸并进行识别
                    face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_cropped = mtcnn(face)
                    if face_cropped is not None:
                        face_cropped = face_cropped
                        face_embedding = resnet(face_cropped.to(device)).cpu().detach().numpy()
                        # 计算特征向量之间的距离，在给定阈值下预测标签，要求能预测出lpr、zcy、zyf
                        distances = np.linalg.norm(embeddings - face_embedding, axis=2)
                        min_distance = np.min(distances)
                        min_index = np.argmin(distances)
                        label = labels[min_index] if min_distance < 0.8 else 'Unknown'
                        # 绘制人脸候选框和标签
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} ({min_distance:.2f})', (int(box[0]), int(box[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # 将处理后的帧编码为JPEG格式
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue  # 如果编码失败，跳过此帧
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # time.sleep(0.01)
    finally:
        cap.release()  # 释放摄像头资源


# 从摄像头获取帧并进行Canny边缘检测
def detect_edges():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue  # 如果读取失败，跳过此帧
            edges = cv2.Canny(frame, 100, 200)  # 进行Canny边缘检测
            ret, jpeg = cv2.imencode('.jpg', edges)
            if not ret:
                continue
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.01)  # 添加适当的延时，以防止发送帧的速度过快
    finally:
        cap.release()


# Flask路由：主页，用于显示实时视频流
@app.route('/')
def index():
    return render_template('index.html')


# Flask路由：用于获取视频流
@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/edge_feed')
def edge_feed():
    return Response(detect_edges(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
