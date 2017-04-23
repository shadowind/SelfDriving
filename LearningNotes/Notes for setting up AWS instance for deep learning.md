## Notes for setting up AWS instance for deep learning  

I've been taken Udacity's self driving car nano degree for 1 month so far. In first semester, we will need to complete 5 projects. They are not easy project just by copy and paste some code and run it. (I could, since there are previous students shared their code on Github). And TAs are very responsible, he gave me very detailed feedback on my projects. And talks to me online every week to see if I have any issue. 

Overall, the experience is great. I did two quite interesting projects already, locally. 

Now here's the problem, my laptop is just not built for such deep learning projects. I waited too long to train one model, and make small tweek parameter and then wait even longer. This largely drag my efficiency, and my desire to work on it. So I decide, (finally), to go with a cloud server.

My IP address: 54.67.64.236
user: carnd
password: carnd
#### Logon to AWS from command line
```
# ssh username@<ip_address>
ssh carnd@54.67.64.236
source activate carnd-term1
```


#### Data transfer between laptop and AWS
```
scp <from_filePath> <to_filePath> # be prompt to type in password
# Upload file
scp <local_dir> carnd@54.67.64.236:<instance directory>
# Download file from AWS
scp carnd@54.67.64.236:<instance directory>  <local_dir>

scp carnd@54.153.95.34:~/CarND-Behavioral-Cloning-P3/best_model.h5 ~/Documents/AA-Study/MOOC/SelfDrive/CarND-Behavioral-Cloning-P3

scp ~/Downloads/Generator_BehaviorClone carnd@52.53.189.95:~/CarND-Behavioral-Cloning-P3/

server_dir: ~/CarND-Behavioral-Cloning-P3/
local_dir: ~/Documents/AA-Study/MOOC/SelfDrive/CarND-Behavioral-Cloning-P3
```
Notice this can only upload file, not folders. So for this reason and faster upload speed. I zip the folder. 

Linux command:  
```
Control + C #Interrupt shell command  
mkdir ./CarND-Behavioral-Cloning-P3 # create directory  
rm -r mydir # Remove directory
```  


### Run model locally
```
python drive.py model.h5
```
[Udacity set up AWS GPU instance guide](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/614d4728-0fad-4c9d-a6c3-23227aef8f66/concepts/f6fccba8-0009-4d05-9356-fae428b6efb4)  
[Amazon EC2 Pricing](https://aws.amazon.com/ec2/pricing/on-demand/) Select region to see pricing  
  
[Machine Learning on Amazon AWS GPU Instances] (https://www.metachris.com/2015/11/machine-learning-on-amazon-aws-gpu-instances/)  
This is a great post about how to lower the cost of machine learning. AWS provide a very smart service called EC2 spot instance. User bid a price, and if there are available instances, you can use it as long as there 
Spot instances get a 2 minute notice before being shut down. You can use boto (AWS SDK for Python) to check the timestamp for when that will occur.

Make sure to snapshot your models, otherwise you might lose training time and have to start over. You can save the snapshots to S3 (depending on the size of the model).

Create an AMI with all dependencies pre-installed so you don’t waste time installing those when the instance spins up.

For very large datasets use their Elastic Block Storage (EBS). It’s basically an on-demand SSD you can attach to instances when they spin up.


