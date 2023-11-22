import tensorflow as tf 
import numpy as np 
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt 
import os ,cv2


"""
Import model definition
"""
from cycleGAN_loss import CycleGAN

"""
Function to load image from path and rescale it to [-1,1]. 
"""
def get_image_new(image_path,width,height):
    image = Image.open(image_path).convert("RGB")
    #image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = cv2.resize(image, (128, 128))
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    image = np.multiply(image,2)    
    return image


"""
Function to test CycleGAN for test images. 
"""
def test(cgan_net,num,testA,testB,model_dir,input_shape,loss_type,image_shape):
    

    path_imgA = model_dir+'/'+testA + '/'
    path_imgB = model_dir+'/'+testB + '/'
    saver = tf.train.Saver()
#    tf.reset_default_graph()
#    tf.train.latest_checkpoint(model_dir)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        print(ckpt.model_checkpoint_path)
        sess.run(tf.global_variables_initializer())
        
        saver.restore(sess,ckpt.model_checkpoint_path)
        for j in range(1,num + 1):
            testA = path_imgA+str(j)+'.png'
            testB = path_imgB+str(j)+'.png'
            imgA = np.reshape(get_image_new(testA,input_shape[0],input_shape[1]),(1,input_shape[0],input_shape[1],input_shape[2]))
            imgB = np.reshape(get_image_new(testB,input_shape[0],input_shape[1]),(1,input_shape[0],input_shape[1],input_shape[2]))
        
            genA,genB,cycA,cycB = sess.run([cgan_net.gen_A,cgan_net.gen_B,cgan_net.cyclicA,cgan_net.cyclicB],
                                       feed_dict={cgan_net.input_A:imgA,cgan_net.input_B:imgB})
            images = [imgA,genB,cycA,imgB,genA,cycB]
            i = 0
            for img in images:
                i = i + 1
                img = np.reshape(img,input_shape)
            


                if np.array_equal(img.max(),img.min()) == False:
                    img = (((img - img.min())*255)/(img.max()-img.min())).astype(np.uint8)
                else:
                    img = ((img - img.min())*255).astype(np.uint8)
                img = Image.fromarray(img)
            #img = plt.imshow(img)
                img = img.resize((image_shape[1], image_shape[0]), Image.ANTIALIAS)
                box = (0, 0, image_shape[1], image_shape[0])
                img = img.crop(box).save('result/'+str(loss_type)+'/'+str(j)+'_'+str(i)+".png")
            #.crop(box)
            
            
            #plt.savefig(str(i)+"test.jpg")
            #ax = plt.axes()
            #ax.grid(False)
            #plt.show()

def main(_):
    print ('begin')
    if not os.path.exists(FLAGS.testA_image):
        print ("TestA image doesn't exist")
    else:
        if not os.path.exists(FLAGS.testB_image):
            print ("TestB image doesn't exist")
        else:
            if not os.path.exists(FLAGS.model_dir):
                print ("CycleGAN model is not available at the specified path")               
            else:
                print ('successful')
                image_shape = [FLAGS.length,FLAGS.width]
                input_shape = 128, 128,3
                batch_size = 1
                pool_size = 50 
                beta1 = 0.5
                loss_type = FLAGS.loss_type
                tf.reset_default_graph()
                
                cgan_net = CycleGAN(batch_size,input_shape,pool_size,beta1,loss_type)
                test(cgan_net,FLAGS.num,FLAGS.testA_image,FLAGS.testB_image,FLAGS.model_dir,input_shape,loss_type,image_shape)       


flags = tf.app.flags
flags.DEFINE_integer("width",64,"Test the width of the image")
flags.DEFINE_integer("length",256,"Test the length of the image")
flags.DEFINE_integer("num",1,"Total number of images")
flags.DEFINE_string("testA_image",None,"TestA Image Path")
flags.DEFINE_string("testB_image",None,"TestB Image Path")
flags.DEFINE_string("model_dir",None,"Path to checkpoint folder")
flags.DEFINE_string("loss_type",None,"Loss type with which cycleGAN model is to be tested")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()
