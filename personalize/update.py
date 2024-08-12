import time
import boto3
import datetime
from common import AWSConfig

personalize = boto3.client("personalize")
aws_config = AWSConfig()
update_date = datetime.date.today()

# 1. data
dataset_import_job_arn_list = []
for data_type in data_types:
    import_mode = "FULL"
        
    update_dataset_import_job_response = personalize.create_dataset_import_job(
        jobName=f"{update_date}_{aws_config.name}_{data_type.upper()}_DATASET_IMPORT",
        datasetArn=f"{aws_config.dataset_arn}/{data_type.upper()}",
        dataSource={
            "dataLocation" : path + data_type + ".csv"
        },
        roleArn=aws_config.role_arn,
        importMode=import_mode,
    )

    dataset_import_job_arn = update_dataset_import_job_response["datasetImportJobArn"]
    dataset_import_job_arn_list.append(dataset_import_job_arn)

max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    for i, data_type in enumerate(data_types):
        describe_dataset_import_job_response = personalize.describe_dataset_import_job(
            datasetImportJobArn = dataset_import_job_arn_list[i]
        )
        status = describe_dataset_import_job_response["datasetImportJob"]["status"]
        print("[{}] DatasetImportJob: {}".format(data_type.upper(), status))

    if status == "ACTIVE" or status == "CREATE FAILED":
        break

    time.sleep(60)
    
# 2. solutions
solution_name_list = []
solution_version_arn_list = []
for solution_name in solution_name_list:
    update_solution_version_update_response = personalize.create_solution_version(
        name = str(update_date),
        solutionArn = f"{aws_config.solution_arn_prefix}/{solution_name}",
        trainingMode = "FULL",
    )
    solution_version_arn = update_solution_version_update_response["solutionVersionArn"]
    solution_version_arn_list.append(solution_version_arn)

max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    for i, solution_version_arn in enumerate(solution_version_arn_list):
        describe_solution_version_response = personalize.describe_solution_version(
            solutionVersionArn = solution_version_arn,
        )
        status = describe_solution_version_response["solutionVersion"]["status"]
        print("[{}] {} SolutionVersion Update: {}".format(solution_name_list[i], update_date, status))

    if status == "ACTIVE" or status == "CREATE FAILED":
        break
    print()

    time.sleep(60)
    
# 3.campaign
campaign_arn_list = []
for solution_name in solution_name_list:
    update_campaign_response = personalize.update_campaign(
        campaignArn=f"{aws_config.campaign_arn_prefix}/{solution_name}_CAMPAIGN",
        solutionVersionArn=f"{aws_config.solution_arn_prefix}/{solution_name}/{update_date}",
        minProvisionedTPS=...,
        campaignConfig={"enableMetadataWithRecommendations":True},
    )
    campaign_arn = update_campaign_response["campaignArn"]
    campaign_arn_list.append(campaign_arn)



max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    for i, campaign_arn in enumerate(campaign_arn_list):
        describe_campaign_response = personalize.describe_campaign(
            campaignArn = campaign_arn,
        )
        status = describe_campaign_response["campaign"]["status"]
        print("[{}] {} Campaign Update: {}".format(solution_name_list[i], update_date, status))

    if status == "ACTIVE" or status == "CREATE FAILED":
        break
    print()

    time.sleep(60)
    
# 4. batch inference job 
update_batch_inference_job_response = personalize.create_batch_inference_job(
    jobName = "",
    solutionVersionArn = personalized_rank_solution_version_arn,
    roleArn = aws_config.role_arn,
    jobInput = {"s3DataSource":{"path":s3_file_in_path}},
    jobOutput = {"s3DataDestination":{"path":s3_file_out_path + "/"}},
    numResults = 1000,
)

total_sec = 0
max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    describe_batch_inference_job_response = personalize.describe_batch_inference_job(
        batchInferenceJobArn = update_batch_inference_job_response["batchInferenceJobArn"],
    )
    status = describe_batch_inference_job_response["batchInferenceJob"]["status"]

    if status == "ACTIVE" or status == "CREATE FAILED":
        break

    time.sleep(30)
    total_sec += 30
