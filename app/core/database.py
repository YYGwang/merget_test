import boto3
from app.core.config import settings

dynamodb = boto3.resource('dynamodb', region_name=settings.REGION)

def get_table(table_name):
    table = dynamodb.Table(table_name)
    return table