import sys
sys.path.insert(0, 'U:/xucheng/yolov8/ultralytics-main')


from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    # model = YOLO("./cfg/models/v8/yolov8-mask-bifpn.yaml")  # 从头开始构建新模型
    model = YOLO("./cfg/models/v8/yolov8-IR-original-gatt-ghost.yaml")  # 从头开始构建新模型


    # model = YOLO("runs/detect/pcbtrain2/weights/best.pt")  # 加载预训练模型（推荐用于训练）

    # Use the model
    #results = model.train(data="./dataset/MaskDataSet/maskdata.yaml", epochs=100, batch=16, workers=8, close_mosaic=0, name='cfg')  # 训练模型
    results = model.train(data="./dataset/HIT_UAV_original/HIT_UAV_original.yaml", epochs=200, batch=16, workers=8, close_mosaic=0, name='HIT_UAV_original_gatt-ghost-train')  # 训练模型

    # results = model.val()  # 在验证集上评估模型性能
    # results = model.predict(source="dataset/PCB/images/test2017/00041131.jpg")  # 预测图像

    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
