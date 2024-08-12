import dataclasses
import datetime
        
        
@dataclasses.dataclass
class AWSConfig:    
    name = ...
    path = ...
    role_arn = ...
    aws_arn_prefix = ...
    dataset_arn = ...
    solution_arn_prefix = ...
    campaign_arn_prefix = ...
    
    def __post_init__(self):
        pass


schema = {
    "interactions": {
        "type": "record",
        "name": "interactions",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {
                ...
            },
        ],
        "version": ""
    },
    
    "users": {
        "type": "record",
        "name": "users",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {
                ...
            }
        ],
        "version": ""
    },
    
    "items": {
        "type": "record",
        "name": "items",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {
                ...
            }
        ],
        "version": ""
    }
}