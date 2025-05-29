import os
import zipfile

zip_name = 'mark-pro.zip'

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        # 跳过隐藏目录和文件
        dirs[:] = [d for d in dirs if not d.startswith('.') or d.startswith('venv') or d == zip_name]
        for file in files:
            if file.startswith('.') or file == 'zip_current_dir.py':
                continue
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, path)
            ziph.write(abs_path, rel_path)

if __name__ == '__main__':
    zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    zipdir('.', zipf)
    zipf.close()