import tensorflow as tf
import os 
import tarfile
import requests
#inception-v3下载地址
url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
dir = 'inception_model'
#检查是否存在目录
if not os.path.exists(dir):
    os.makedirs(dir)
filename = url.split('/')[-1]
filepath = os.path.join(dir,filename)
#发起请求，并下载文件
if not os.path.exists(filepath):
    print("download ", filename)
    r = requests.get(url,stream = True)
    with open(filepath,'wb') as f:
        for chunk in r.iter_content(chunk_size = 1024):
            if chunk:
                f.write(chunk)
print("finish ",filename)
#解压文件
tarfile.open(filepath,'r:gz').extractall(dir)
log_dir = 'inception_log'
if not os.path.exists(log_dir):
#    os.mkdir(log_dir)
    os.makedirs(log_dir)
#pb格式为下载的模型，需要从其中提取并保存网络参数
graph_file = os.path.join(dir,'classify_image_graph_def.pb')
with tf.Session() as sess:
    #定义图，将pb的参数保存在图中
    with tf.gfile.FastGFile(graph_file,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def=graph_def,name="inception-v3")
    #将获取的参数本地化存储
    writer = tf.summary.FileWriter(log_dir,sess.graph)
    writer.close()
    
