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
def get_data(trainA,trainB,width,height):
  tr_A = []
  tr_B = []
  for i in range(1,4501):
    tr_A.append(get_image_new("Dataset//trainA//"+str(i)+'.png',128,128))
    if i % 200 == 0:
      print ("getting trainA = %r" %(i))
  for i in range(1,4501):
    tr_B.append(get_image_new("Dataset//trainB//"+str(i)+'.png',128,128))
    if i % 200 == 0:
      print ("getting trainB = %r" %(i))

  tr_A = np.array(tr_A)
  tr_B = np.array(tr_B)
  print ("Completed loading training data. DomainA = %r , DomainB = %r" %(tr_A.shape,tr_B.shape))
  return tr_A,tr_B


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
Function to train the network
"""
def evaluation(high,wide,img1,img2,batch_size):
    #print(img1.shape,img2.shape)
    #img1 = np.squeeze(img1)
    #img2 = np.squeeze(img2)
    #print(img1.shape,img2.shape)
    #img1 = Image.fromarray(img1)
    #img2 = Image.fromarray(img2)
    TP = 0
    FP = 0
    FN = 0
    #img1 = img1.convert('L')
    #img2 = img2.convert('L')
    img1 = np.array(img1)
    img2 = np.array(img2)
    for l in range(batch_size):
        for j in range(wide):
            for k in range(high):
                if img1[0,k, j,0] - img2[0,k, j,0] >1:
                    FP = FP + 1               
                else:
                    if img1[0,k, j,0] - img2[0,k, j,0] <-1:                    
                        FN = FN + 1
                    else:
                        TP = TP + 1
 #   print(TP / (TP + FP + FN))
#    print((TP * 2) / (TP * 2 + FP + FN))
    return TP,FP,FN

def calculate(TP,FP,FN):
    F1 = (TP * 2) / (TP * 2 + FP + FN)
    IOU = TP / (TP + FP + FN)
    return F1,IOU

def train(cgan_net,max_img,batch_size,trainA,trainB,lr_rate,shape,pool_size,epoch,model_dir):
    lr_init = lr_rate
    pool_size = int(pool_size / batch_size)
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
    start_epoch = 0


    allTime = 0
    poolA = np.zeros((pool_size,batch_size,shape[0],shape[1],shape[2]))
    poolB = np.zeros((pool_size,batch_size,shape[0],shape[1],shape[2]))
    #计时开始
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while start_epoch < epoch:
            start = time.perf_counter()#逐代计时
            if start_epoch >= 100:
                lr_rate = lr_init - ((start_epoch-100)*lr_init)/100
            countA = 0
            countB = 0            
            np.random.shuffle(trainA)                
            np.random.shuffle(trainB)
            
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
                
                #poolA,poolB,num_im = save_to_pool(poolA,poolB,genA,genB,pool_size,num_im)
                num_im = step - 1
                poolA[num_im] = genA
                poolB[num_im] = genB
                
                if num_im >=1:
                    indA = random.randint(0,num_im)
                    indB = random.randint(0,num_im)
                else:
                    indA = 0
                    indB = 0
                fakeA_img = poolA[indA]
                fakeB_img = poolB[indB]
                #逐代计算 f1，miou
                #tp,fp,fn = evaluation(128,128,imgA,fakeA_img)
                #tp,fp,fn = evaluation(128,128,imgA,fakeB_img)
                tp,fp,fn = evaluation(128,128,imgB,fakeB_img,batch_size)
                TP = TP + tp
                FP = FP + fp
                FN = FN + fn
                _,discA_loss,_,discB_loss = sess.run([cgan_net.discA_opt,cgan_net.disc_loss_A,cgan_net.discB_opt,cgan_net.disc_loss_B],
                         feed_dict={cgan_net.input_A:imgA,cgan_net.input_B:imgB,cgan_net.lr_rate:lr_rate,cgan_net.fake_pool_Aimg:fakeA_img,cgan_net.fake_pool_Bimg:fakeB_img})
                
                #移到下一行
                #tlogging training loss details 
                #if step % 1500 == 0 or epoch % 5 == 0:
                    #print ("epoch = %r step = %r discA_loss = %r genA_loss = %r discB_loss = %r genB_loss = %r" 
                    #      %(epoch,step,discA_loss,genA_loss,discB_loss,genB_loss))
            #逐代f1，miou
            #逐代计时结束             
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
                           %(start_epoch,step+1,f1,miou,discA_loss,genA_loss,discB_loss,genB_loss,runTime))
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
        #打印各类图表
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
    
        trainA = glob(FLAGS.data_path+"//trainA//"+FLAGS.input_fname_pattern)
        trainB = glob(FLAGS.data_path+"//trainB//"+FLAGS.input_fname_pattern)
        tr_imgA, tr_imgB = get_data(trainA,trainB,128,128)
        
        #超参数
        input_shape = 128,128,3
        batch_size = FLAGS.batch_size
        pool_size = 4500
        print(batch_size)
        lr_rate = 0.0002
        beta1 = 0.5
        max_img = 4500
        # change loss type. Options - l1, l2, ssim, ssim_l1, ssim_l2_a, ssim_l2_b, ssim_l1l2_a, ssim_l1l2_b, l1_l2
        loss_type = FLAGS.loss_type
        epoch = FLAGS.epoch
        tf.reset_default_graph()
        
        cgan_net = CycleGAN(batch_size,input_shape,pool_size,beta1,loss_type)
        
        train(cgan_net, max_img, batch_size, tr_imgA, tr_imgB, lr_rate, input_shape, pool_size, epoch, os.path.join(FLAGS.model_dir, 'model_' + loss_type))



flags = tf.app.flags
flags.DEFINE_string("data_path",None,"Path to parent directory of trainA and trainB folder")
flags.DEFINE_string("input_fname_pattern","*.jpg","Glob pattern of training images")
flags.DEFINE_string("model_dir","CycleGAN_model","Directory name to save checkpoints")
flags.DEFINE_string("loss_type","l1","Loss type with which cycleGAN is to be trained")
flags.DEFINE_integer("batch_size", 1,"Enhace the speed of cycleGAN")
flags.DEFINE_integer("epoch", 200,"The epoch of cycleGAN")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()

