# 从服务器上下载数据文件
import ftplib
import os

# FTP服务器地址
FTP_HOST = 'ftp.example.com'
# FTP用户名
FTP_USER = 'your_username'
# FTP密码
FTP_PASS = 'your_password'
# 远程文件路径
REMOTE_PATH = '/path/to/remote/files'
# 本地保存路径
LOCAL_PATH = '/path/to/local/directory'

# 连接FTP服务器
ftp = ftplib.FTP(FTP_HOST)
ftp.login(FTP_USER, FTP_PASS)

# 切换到远程文件目录
ftp.cwd(REMOTE_PATH)

# 获取远程目录下所有文件名
files = ftp.nlst()

# 遍历文件并下载
for filename in files:
    # 根据您的规则判断是否需要下载该文件
    if should_download(filename):
        local_filename = os.path.join(LOCAL_PATH, filename)
        with open(local_filename, 'wb') as file:
            try:
                ftp.retrbinary(f'RETR {filename}', file.write)
                print(f'Downloaded: {filename}')
            except ftplib.error_perm as e:
                print(f'Error downloading {filename}: {e}')

# 关闭FTP连接
ftp.quit()