import tensorflow as tf 
import numpy as np 
from glob import glob
import random
from PIL import Image 
import os,time
import matplotlib.pyplot as plt

"""
Import CycleGAN class definition.
"""
from cycleGAN_loss import CycleGAN

"""
Function to draw picture. 
"""
def draw_relation(epoch,val,name):
    x = epoch
    y = val
    plt.figure()

    plt.plot(x, y ,color='b')
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.savefig('result/'+name+'.png')
   

"""
Function to load image and rescale it to [-1,1].
"""
def get_image_new(image_path,width,height):
    image = Image.open(image_path).convert("RGB")    #image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    image = np.multiply(image,2)    
    return image

"""
Function to load training images.
"""
def get_data(data_path,mod,num):
  tr_A = []
  tr_B = []
  for i in range(1,num + 1):
    tr_A.append(get_image_new(data_path + "//trainA//"+str(i)+mod,128,128))
    if i % 200 == 0:
      print ("getting trainA = %r" %(i))
  for i in range(1,num + 1):
    tr_B.append(get_image_new(data_path + "//trainB//"+str(i)+mod,128,128))
    if i % 200 == 0:
      print ("getting trainB = %r" %(i))

  tr_A = np.array(tr_A)
  tr_B = np.array(tr_B)
  print ("Completed loading training data. DomainA = %r , DomainB = %r" %(tr_A.shape,tr_B.shape))
  return tr_A,tr_B


"""
Function to save generated image to image pools. 
"""
def save_to_pool(poolA,poolB,gen_A,gen_B,pool_size,num_im):
        
        if num_im < pool_size:
            poolA[num_im] = gen_A
            poolB[num_im] = gen_B
            num_im = num_im + 1
        
        else:
            p = random.random()
            if p > 0.5:
                indA = random.randint(0,pool_size-1)
                poolA[indA] = gen_A
            p = random.random()
            if p > 0.5: 
                indB = random.randint(0,pool_size-1)
                poolB[indB] = gen_B
                
        return poolA,poolB,num_im

"""
Function to calculate confusion matrix.
"""
def evaluation(high,wide,img1,img2):
    TP = 0
    FP = 0
    FN = 0
    img1 = np.array(img1)
    img2 = np.array(img2)
    for j in range(wide):
        for k in range(high):
            if img1[0,k, j,0] - img2[0,k, j,0] >1:
                FP = FP + 1               
            else:
                if img1[0,k, j,0] - img2[0,k, j,0] <-1:                    
                    FN = FN + 1
                else:
                    TP = TP + 1
    return TP,FP,FN

"""
Function to calculate evaluation indicators.
"""
def calculate(TP,FP,FN):
    F1 = (TP * 2) / (TP * 2 + FP + FN)
    IOU = TP / (TP + FP + FN)
    return F1,IOU

def train(cgan_net,max_img,batch_size,trainA,trainB,lr_rate,shape,pool_size,model_dir,epoch):
    lr_init = lr_rate
    Time = []
    F1 = []
    mIOU = []
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    saver = tf.train.Saver(max_to_keep=None)
    lenA = len(trainA)
    lenB = len(trainB)
    epoch = 0
    countA = 0 
    countB = 0
    allTime = 0
    poolA = np.zeros((pool_size,1,shape[0],shape[1],shape[2]))
    poolB = np.zeros((pool_size,1,shape[0],shape[1],shape[2]))
    #计时开始
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while start_epoch < epoch:
            start = time.perf_counter()
            if start_epoch >= 100:
                lr_rate = lr_init - ((start_epoch-100)*lr_init)/100
            countA = 0
            countB = 0
            
            for step in range(pool_size):
                
                TP = 0
                FP = 0
                FN = 0
                
                imgA = []
                imgB = []
                for count in range(0, batch_size):
                    
                    imgA.append(trainA[countA])
                    imgB.append(trainB[countB])
                    testImg = imgA
                    countA = countA + 1
                    countB = countB + 1
                
                imgA = np.reshape(imgA,(batch_size,shape[0],shape[1],shape[2]))
                imgB = np.reshape(imgB,(batch_size,shape[0],shape[1],shape[2]))
               
                
                _,genB,genA_loss,_,genA,genB_loss,cyclicA,cyclicB = sess.run([cgan_net.genA_opt,cgan_net.gen_B,cgan_net.gen_loss_A,cgan_net.genB_opt,cgan_net.gen_A,cgan_net.gen_loss_B,cgan_net.cyclicA,cgan_net.cyclicB],
                                            feed_dict={cgan_net.input_A:imgA,cgan_net.input_B:imgB,cgan_net.lr_rate:lr_rate})

                poolA[countA - 1] = genA
                poolB[countB - 1] = genB
                num_im = countA - 1

                fakeA_img = poolA[num_im]
                fakeB_img = poolB[num_im]

                tp,fp,fn = evaluation(128,128,imgB,fakeB_img)
                TP = TP + tp
                FP = FP + fp
                FN = FN + fn
                _,discA_loss,_,discB_loss = sess.run([cgan_net.discA_opt,cgan_net.disc_loss_A,cgan_net.discB_opt,cgan_net.disc_loss_B],
                         feed_dict={cgan_net.input_A:imgA,cgan_net.input_B:imgB,cgan_net.lr_rate:lr_rate,cgan_net.fake_pool_Aimg:fakeA_img,cgan_net.fake_pool_Bimg:fakeB_img})
                
           
            f1,miou = calculate(TP,FP,FN)
            end = time.perf_counter()
            runTime = end - start
            F1.append(f1)
            mIOU.append(miou)
            Time.append(runTime)
            loss1.append(discA_loss)
            loss2.append(genA_loss)
            loss3.append(discB_loss)
            loss4.append(genB_loss)
            allTime = allTime + runTime
            start_epoch = start_epoch + 1
            print ("epoch = %r step = %r F1 = %r mIOU = %r discA_loss = %r genA_loss = %r discB_loss = %r genB_loss = %r runTime = %r" 
                           %(epoch,step+1,f1,miou,discA_loss,genA_loss,discB_loss,genB_loss,runTime))
            if start_epoch % 20 == 0:
                saver.save(sess,model_dir,write_meta_graph=True)
                print ("### Model weights Saved epoch = %r ###" %(epoch))
        
        saver.save(sess,model_dir,write_meta_graph=True)
        print("whole time is %r"%(allTime))
        print ("### Model weights Saved epoch = %r ###" %(epoch))
        file = open('result.txt','w')
        file.write(str(F1));
        file.write(str(mIOU));
        file.write(str(time));
        file.write(str(discA_loss));
        file.write(str(genA_loss));
        file.write(str(discB_loss));
        file.write(str(genB_loss));
        file.write(str(allTime));
        file.close()
        axis_x = list(range(1,epoch+1))
        draw_relation(axis_x,F1,'F1')
        draw_relation(axis_x,mIOU,'mIOU')
        draw_relation(axis_x,Time,'time')
        draw_relation(axis_x,loss1,'discA_loss')
        draw_relation(axis_x,loss2,'genA_loss')
        draw_relation(axis_x,loss3,'discB_loss')
        draw_relation(axis_x,loss4,'genB_loss')
        


def main(_): 
    if not os.path.exists(FLAGS.data_path):
        print ("Training Path doesn't exist")
    else:
            
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)

        tr_imgA, tr_imgB = get_data(FLAGS.data_path,FLAGS.input_fname_pattern,FLAGS.num)
        
        input_shape = 128,128,3
        batch_size = FLAGS.batch_size
        pool_size = FLAGS.num
        print(batch_size)
        lr_rate = 0.0002
        beta1 = 0.5
        max_img = FLAGS.num
        loss_type = FLAGS.loss_type
        epoch = FLAGS.epoch
        tf.reset_default_graph()
        
        cgan_net = CycleGAN(batch_size,input_shape,pool_size,beta1,loss_type)
        
        train(cgan_net, max_img, batch_size, tr_imgA, tr_imgB, lr_rate, input_shape, pool_size, FLAGS.epoch, os.path.join(FLAGS.model_dir, 'model_' + loss_type))



flags = tf.app.flags
flags.DEFINE_string("data_path",None,"Path to parent directory of trainA and trainB folder")
flags.DEFINE_string("input_fname_pattern",".jpg","Glob pattern of training images")
flags.DEFINE_string("model_dir","CycleGAN_model","Directory name to save checkpoints")
flags.DEFINE_string("loss_type","l1","Loss type with which cycleGAN is to be trained")
flags.DEFINE_integer("batch_size", 1,"Enhace the speed of cycleGAN")
flags.DEFINE_integer("epoch", 200,"The epoch of cycleGAN")
flags.DEFINE_integer("num",1,"Total number of training images")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()


