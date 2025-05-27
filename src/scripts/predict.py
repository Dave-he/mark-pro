def batch_predict():
    # 设备配置从配置文件读取
    device = torch.device(cfg.train.device if hasattr(cfg.train, 'device') else 
                   cfg.device if hasattr(cfg, 'device') else 
                   'cuda' if torch.cuda.is_available() else 'cpu')
    
    # 记录设备信息
    print(f"使用设备: {device}")
    
    # 加载模型
    model = SegGuidedUnetPP().to(device)
    
    # 尝试加载最佳模型，如果失败则尝试加载最新模型
    try:
        imgsize = f"{cfg.data.image_size[0]}x{cfg.data.image_size[1]}"
        segw = cfg.train.seg_loss_weight
        model_name = f"unetpp_img{imgsize}_sw{segw}.pth"
        model_path = os.path.join(cfg.model.save_dir, model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"加载模型: {model_path}")
    except Exception as e:
        print(f"无法加载指定模型: {str(e)}")
        # 尝试加载最新模型
        model_files = [f for f in os.listdir(cfg.model.save_dir) if f.endswith('.pth')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(cfg.model.save_dir, latest_model)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"加载最新模型: {model_path}")
        else:
            raise FileNotFoundError("找不到任何模型文件")
    
    # 创建输出目录
    os.makedirs(cfg.predict.output_dir, exist_ok=True)
    
    # 获取所有输入图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(cfg.predict.input_dir) 
                  if os.path.isfile(os.path.join(cfg.predict.input_dir, f)) 
                  and os.path.splitext(f)[1].lower() in image_extensions]
    
    # 批量预测
    from tqdm import tqdm
    for i, filename in enumerate(tqdm(image_files, desc="处理图像")):
        image_path = os.path.join(cfg.predict.input_dir, filename)
        
        try:
            predict_single_image(model, image_path, cfg.predict.output_dir, visualize=False)
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
    
    print(f"所有图像处理完成。结果保存至 {cfg.predict.output_dir}")