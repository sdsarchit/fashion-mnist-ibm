#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade ibm-cos-sdk')


# In[2]:


import ibm_boto3
from ibm_botocore.client import Config


# In[5]:


cos_credentials = {
  "apikey": "OwHl-WobDNvVVuUtjNbNK4DRHYfI8UF2svbIfcXS8_vi",
  "cos_hmac_keys": {
    "access_key_id": "0a2c727131a34ec5825b614b93fd5d27",
    "secret_access_key": "89fd2f302037fac154b858ed20928dcc4d6e971fd81c27aa"
  },
  "endpoints": "https://control.cloud-object-storage.cloud.ibm.com/v2/endpoints",
  "iam_apikey_description": "Auto-generated for key 0a2c7271-31a3-4ec5-825b-614b93fd5d27",
  "iam_apikey_name": "Service credentials-1",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/3d9b61ef537f4e728670933348540cb3::serviceid:ServiceId-346be476-38c1-4dbd-8355-483bb769ce71",
  "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/3d9b61ef537f4e728670933348540cb3:894f52ce-f97f-444f-852c-82a4a950e182::"
}


# In[10]:


# Define endpoint information.
service_endpoint = 'https://s3-api.us-geo.objectstorage.softlayer.net'


# In[8]:


auth_endpoint = 'https://iam.bluemix.net/oidc/token'


# In[11]:


# Create a COS resource.
cos = ibm_boto3.resource(
    's3',
     ibm_api_key_id=cos_credentials['apikey'],
     ibm_service_instance_id=cos_credentials['resource_instance_id'],
     ibm_auth_endpoint=auth_endpoint,
     config=Config(signature_version='oauth'),
     endpoint_url=service_endpoint
)


# In[13]:


from uuid import uuid4

bucket_uid = str(uuid4())
buckets = ['gan-training-data-' + bucket_uid, 'gan-training-results-' + bucket_uid]

for bucket in buckets:
    if not cos.Bucket(bucket) in cos.buckets.all():
        print('Creating bucket "{}"...'.format(bucket))
        try:
            cos.create_bucket(Bucket=bucket)
        except ibm_boto3.exceptions.ibm_botocore.client.ClientError as e:
            print('Error: {}.'.format(e.response['Error']['Message']))


# In[14]:


print([x.name for x in list(cos.buckets.all()) if x.name.startswith('gan-training')])


# In[15]:


data_links = [
    'https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-images-idx3-ubyte.gz?raw=true' # Training set images
]


# In[16]:


from urllib.request import urlopen

bucket_obj = cos.Bucket(buckets[0])

for data_link in data_links:
    filename = data_link.split('/')[-1].split('?')[0]
    print('Uploading data {}...'.format(filename))
    with urlopen(data_link) as data:
        bucket_obj.upload_fileobj(data, filename)
        print('{} is uploaded.'.format(filename))


# In[17]:


for bucket_name in buckets:
    print(bucket_name)
    bucket_obj = cos.Bucket(bucket_name)
    for obj in bucket_obj.objects.all():
        print('\tFile: {}, {:4.2f} kB'.format(obj.key, obj.size/1024))


# In[18]:


get_ipython().system('pip install --upgrade wget')


# In[19]:


import json
import os
import wget


# In[29]:


wml_credentials = {
  "apikey": "uW7C5QsAFJ0ye3Bg460wwiMLQ87lsrnQMWm9eOV5DzTC",
  "iam_apikey_description": "Auto-generated for key a4d49afc-30f8-462e-bb3a-fe3fac2d43ee",
  "iam_apikey_name": "Service credentials-2",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/3d9b61ef537f4e728670933348540cb3::serviceid:ServiceId-0a9b8e27-1147-4ff8-82c0-21d5302f018f",
  "instance_id": "87c1a9b2-fb81-4f41-a9de-07b9d2cad65c",
  "url": "https://us-south.ml.cloud.ibm.com"
}


# In[25]:


get_ipython().system('rm -rf $PIP_BUILD/watson-machine-learning-client')


# In[22]:


get_ipython().system('pip install --upgrade watson-machine-learning-client')


# In[30]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient

client = WatsonMachineLearningAPIClient(wml_credentials)


# In[31]:


from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np


# In[32]:


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# In[33]:


print('Train data set dimension: {}'.format(X_train.shape))
print('Test data set dimension: {}'.format(X_test.shape))
print('Train label dimension: {}'.format(y_train.shape))
print('Test label dimension: {}'.format(y_test.shape))


# In[34]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_images(X_train, y_train, class_names, row_num, col_num, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    for i in range(row_num * col_num):
        plt.subplot(row_num, col_num, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])


# In[37]:


plot_images(X_train,y_train,class_names,6,6)


# In[38]:


model_definition_metadata = {
    client.repository.DefinitionMetaNames.NAME: 'gan_training',
    client.repository.DefinitionMetaNames.DESCRIPTION: 'gan_training-definition',
    client.repository.DefinitionMetaNames.AUTHOR_NAME: 'Jihyoung Kim',
    client.repository.DefinitionMetaNames.FRAMEWORK_NAME: 'tensorflow',
    client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: '1.13',
    client.repository.DefinitionMetaNames.RUNTIME_NAME: 'python',
    client.repository.DefinitionMetaNames.RUNTIME_VERSION: '3.6',
    client.repository.DefinitionMetaNames.EXECUTION_COMMAND: 'python3 gan_fashion_mnist.py --epochs 30000'
}


# In[39]:


filename = 'gan_fashion_mnist.zip'

if not os.path.isfile(filename):
    filename = wget.download('https://github.com/IBMDataScience/sample-notebooks/raw/master/Files/gan_fashion_mnist.zip')
    
print(filename)


# In[40]:


definition_details = client.repository.store_definition(filename, model_definition_metadata)
definition_uid = client.repository.get_definition_uid(definition_details)
print(definition_uid)


# In[48]:


training_configuration_metadata = {
    client.training.ConfigurationMetaNames.NAME: 'gan_training', 
    client.training.ConfigurationMetaNames.AUTHOR_NAME: 'Jihyoung Kim',              
    client.training.ConfigurationMetaNames.DESCRIPTION: 'gan_training_definition',
    client.training.ConfigurationMetaNames.COMPUTE_CONFIGURATION: {'name': 'k80'},
    client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCE: {
        'connection': {
            'endpoint_url': service_endpoint,
            'access_key_id': cos_credentials['cos_hmac_keys']['access_key_id'],
            'secret_access_key': cos_credentials['cos_hmac_keys']['secret_access_key']
        },
        'source': {
            'bucket': buckets[0]
        },
        'type': 's3'
    },
    client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
        'connection': {
            'endpoint_url': service_endpoint,
            'access_key_id': cos_credentials['cos_hmac_keys']['access_key_id'],
            'secret_access_key': cos_credentials['cos_hmac_keys']['secret_access_key']
        },
        'target': {
            'bucket': buckets[1]
        },
        'type': 's3'
    }
}


# In[49]:


training_run_details = client.training.run(definition_uid, training_configuration_metadata)


# In[50]:


training_run_guid_async = client.training.get_run_uid(training_run_details)


# In[51]:


status = client.training.get_status(training_run_guid_async)
print(json.dumps(status, indent=2))


# In[53]:


from time import time
ts=time()
client.training.monitor_logs(training_run_guid_async)
te=time()


# In[54]:


print('Time elapsed: {:.2f} min'.format((te - ts) / 60))


# In[55]:


saved_model_details = client.repository.store_model(
    training_run_guid_async, {'name': 'Fashion MNIST GAN model'}
)


# In[56]:


print('Url: {}'.format(client.repository.get_model_url(saved_model_details)))


# In[57]:


model_uid = client.repository.get_model_uid(saved_model_details)
print('Saved model uid: {}'.format(model_uid))


# In[58]:


deployment_details = client.deployments.create(model_uid, 'Fashion MNIST GAN model deployment')


# In[59]:


scoring_url = client.deployments.get_scoring_url(deployment_details)
print(scoring_url)


# In[65]:


col_num = 6
row_num = 6
payload = np.random.normal(0, 1, (row_num * col_num, 100))
payload.shape


# In[66]:


pred = client.deployments.score(scoring_url, {'values': payload.tolist()})
generated_images=0.5*np.array(pred['values'])+0.5


# In[67]:


plt.figure(figsize=(10, 10))
for i in range(row_num * col_num):
    plt.subplot(row_num, col_num, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(generated_images[i, :, :, 0], cmap=plt.cm.binary)


# In[ ]:




