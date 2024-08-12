import json
import time
import boto3
import datetime
import numpy as np
import pandas as pd
import awswrangler as wr
import dataclasses

from tqdm.auto import tqdm
from common import AWSConfig


aws_config = AWSConfig()
session = boto3.Session(profile_name="")
personalize = session.client("personalize")


# 1. 데이터셋 그룹 생성
create_dataset_group_response = personalize.create_dataset_group(
    name = aws_config.name,
)
dataset_group_arn = create_dataset_group_response["datasetGroupArn"]
max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    describe_dataset_group_response = personalize.describe_dataset_group(
        datasetGroupArn = dataset_group_arn,
    )
    status = describe_dataset_group_response["datasetGroup"]["status"]
    print("Dataset Group: {}".format(status))

    if status == "ACTIVE" or status == "CREATE FAILED":
        break

    time.sleep(30)

for data_type in data_types: # items
    ###########################################################################
    # 2. 각 schema 정의
    try:
        create_schema_response = personalize.create_schema(
            name = f"{aws_config.name}_{data_type.upper()}_SCHEMA",
            schema = json.dumps(schema[data_type]),
        )
    except:
        print("already exist")
    ###########################################################################
    
    
    ###########################################################################
    # 3. 각 dataset 정의
    try:
        create_dataset_response = personalize.create_dataset(
            name = f"{aws_config.name}_{data_type.upper()}_DATASET",
            datasetType = data_type,
            datasetGroupArn = dataset_group_arn,
            schemaArn = create_schema_response["schemaArn"],
        )
    except:
        print("already exist")
    max_time = time.time() + 3*60*60 # 3 hours
    while time.time() < max_time:
        describe_dataset_response = personalize.describe_dataset(
            datasetArn = create_dataset_response["datasetArn"],
        )
        status = describe_dataset_response["dataset"]["status"]
        print("[{}] Dataset: {}".format(data_type.upper(), status))

        if status == "ACTIVE" or status == "CREATE FAILED":
            break

        time.sleep(60)
    
    ###########################################################################
    
    
    ###########################################################################
    # 4. dataset에 import job 정의
    create_dataset_import_job_response = personalize.create_dataset_import_job(
        jobName = f"{today}_{aws_config.name}_{data_type.upper()}_DATASET_IMPORT",
        datasetArn = create_dataset_response["datasetArn"],
        dataSource = {
            "dataLocation": path + data_type + ".csv" 
        },
        roleArn = aws_config.role_arn,
        importMode = "FULL",
    )
    max_time = time.time() + 3*60*60 # 3 hours
    while time.time() < max_time:
        describe_dataset_import_job_response = personalize.describe_dataset_import_job(
            datasetImportJobArn = create_dataset_import_job_response["datasetImportJobArn"],
        )
        status = describe_dataset_import_job_response["datasetImportJob"]["status"]
        print("[{}] DatasetImportJob: {}".format(data_type.upper(), status))

        if status == "ACTIVE" or status == "CREATE FAILED":
            break

        time.sleep(60)
    ###########################################################################
    
    
solution_name_list = []
recipe_arn_list = [] 
solution_version_arn_list = []
for solution_name, recipe_arn in zip(solution_name_list, recipe_arn_list):
    create_solution_response = personalize.create_solution(
        name = solution_name,
        recipeArn = recipe_arn,
        datasetGroupArn = dataset_group_arn,
    )
    solution_arn = create_solution_response["solutionArn"]

    create_solution_version_response = personalize.create_solution_version(
        name = str(datetime.date.today()), # solution version ID
        solutionArn = solution_arn,
        trainingMode = "FULL",
    )
    solution_version_arn = create_solution_version_response["solutionVersionArn"]
    solution_version_arn_list.append(solution_version_arn)
    
max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    for i, solution_version_arn in enumerate(solution_version_arn_list):
        describe_solution_version_response = personalize.describe_solution_version(
            solutionVersionArn = solution_version_arn
        )
        status = describe_solution_version_response["solutionVersion"]["status"]
        print("[{}] SolutionVersion: {}".format(solution_name_list[i], status))

    if status == "ACTIVE" or status == "CREATE FAILED":
        break

    time.sleep(60)
    
campaign_arn_list = []
for i, solution_version_arn in enumerate(solution_version_arn_list):
    create_campaign_response = personalize.create_campaign(
        name = solution_name_list[i] + "_CAMPAIGN",
        solutionVersionArn = solution_version_arn,
        minProvisionedTPS = ..., # the minimum provisioned transactions per second
        campaignConfig = {"enableMetadataWithRecommendations":True}
    )
    campaign_arn = create_campaign_response["campaignArn"]
    campaign_arn_list.append(campaign_arn)

max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    for i, campaign_arn in enumerate(campaign_arn_list):
        describe_campaign_response = personalize.describe_campaign(
            campaignArn = campaign_arn
        )
        status = describe_campaign_response["campaign"]["status"]
        print("[{}] Campaign: {}".format(solution_name_list[i]+"_CAMPAIGN", status))

        if status == "ACTIVE" or status == "CREATE FAILED":
            break

    time.sleep(60)
    


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