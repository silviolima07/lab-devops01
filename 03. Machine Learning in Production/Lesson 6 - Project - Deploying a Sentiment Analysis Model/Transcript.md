## Deployment Project

[Video](https://youtu.be/LWcJtUKVkzo)
'---


## Setting up a Notebook Instance

The deployment project which you will be working on is intended to be done using Amazon's SageMaker platform. In particular, it is assumed that you have a working notebook instance in which you can clone the deployment repository.

If you have not yet done this, please see the beginning of Lesson 2 in which we walk through creating a notebook instance and cloning the deployment repository. Alternatively, you can follow the instructions below.

First, start by logging in to the AWS console, opening the SageMaker dashboard and clicking on Create notebook instance.

You may choose any name you would like for your notebook. A ml.t2.medium is used to launch the notebook and is available by default. Inside the notebook ml.p2.xlarge is used for training a model and ml.m4.xlarge is used for deployment. These instance may not be available to all users by default. If you haven't requested ml.p2.xlarge so far please follow the instructions on the next page to request it now.

Next, under IAM role select Create a new role. You should get a pop-up window that looks like the one below. The only change that needs to be made is to select None under S3 buckets you specify, as is shown in the image below.
Create an IAM role dialog box

Once you have finished setting up the role for your notebook, your notebook instance settings should look something like the image below.
Notebook instance settings

Note that your notebook name may be different than the one displayed and the IAM role that appears will be different.

Next, scroll down to the section labelled Git repositories. Here you will clone the https://github.com/udacity/sagemaker-deployment.git repository.

Once you have filled in all of the required values, the settings should look as so:

You're done! Click on Create notebook instance.

Your notebook instance is now set up and ready to be used!

Once the Notebook instance has loaded, you will see a screen resembling the following.

You can access your notebook using the Action "Open Jupyter".

---

## A. AWS Service Utilization Quota (Limits)

You need to understand the way AWS imposes utilization quotas (limits) on almost all of its services. Quotas, also referred to as limits, are the maximum number of resources of a particular service that you can create in your AWS account.

- AWS provides default quotas, **for each AWS service.**
- Importantly, **each quota is region-specific**.
- There are three ways to **view your quotas**, as mentioned here:
  1. Service Endpoints and Quotas,
  2. Service Quotas console,
  3.  AWS CLI commands - ``list-service-quota``s and ``list-aws-default-service-quotas``
- In general, there are three ways to **increase the quotas**:
  1.  Using [Amazon Service Quotas](https://aws.amazon.com/about-aws/whats-new/2019/06/introducing-service-quotas-view-and-manage-quotas-for-aws-services-from-one-location/) service - This service consolidates your account-specific values for quotas across all AWS services for improved manageability. Service Quotas is available at no additional charge. You can directly try logging into [Service Quotas console](https://console.aws.amazon.com/servicequotas/home) here.
  2. Using [AWS Support Center](https://console.aws.amazon.com/support/home) - You can create a case for support from AWS.
  3. AWS CLI commands - ``request-service-quota-increase``

## A.1. Amazon SageMaker Utilization Quota (Limits)

You can view the Amazon SageMaker Service Limits at ["Amazon SageMaker Endpoints and Quotas"](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html) page. You can request to increase the AWS Sagemaker quota using the [*AWS Support Center*](https://console.aws.amazon.com/support/home) only. Note that ``currently the *Amazon Service Quotas* does not support SageMaker service``. However, SageMaker would be introduced soon into Service Quotas. AWS is moving to make users manage quotas for all AWS services from one central location.

SageMaker would be introduced soon into Services Quota - Courtesy - [Amazon Service Quotas](https://aws.amazon.com/about-aws/whats-new/2019/06/introducing-service-quotas-view-and-manage-quotas-for-aws-services-from-one-location/)

## A.2. Sagemaker Instance Quota (Limit)

Udacity has already set increased quotas for ml.m4.xlargeand ml.p2.xlarge so students won't need to worry about these quotas during the duration of their coursework.

1. Sign in to AWS console - https://aws.amazon.com/console/

Sign in to AWS console

2. Go to the [AWS Support Center(https://console.aws.amazon.com/support/home#/)] and create a case.

AWS Support Center

3. Click on *Service limit increase*

Create a case for support

4. It will expand three sections - Case classification, Case description, and Contact options on the same page. In Case classification section, select "Sagemaker" as the Limit type.

Case classification section that takes the Limit type

5. It will expand one more section - Requests on the same page. In Request section, and select the Region in which you are using the SageMaker service.
   1. Select Sagemaker Training as the Resource Type
   2. Select the instance type (ml.m4.xlarge or ml.p2.xlarge) under the Limit field
   3. Under new limit value, select 1

Request section that takes Region, Resource type, and LImit values

6. Provide a case description and the contact options before submitting the case to support.

    IMPORTANT NOTICE: ``This is the current AWS UI as of April 6th, 2020. The AWS UI is subject to change on a regular basis. We advise students to refer to AWS documentation for the above process.``

