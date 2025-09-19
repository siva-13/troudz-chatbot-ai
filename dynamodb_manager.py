"""
DynamoDB Manager for RAG Application - FIXED VERSION
Handles user profiles and conversation persistence
"""

import boto3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from botocore.exceptions import ClientError
import json
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamoDBManager:
    def __init__(self, region_name: str = 'us-east-1', profile_name: str = None):
        """Initialize DynamoDB manager"""
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
            self.dynamodb = session.resource('dynamodb', region_name=region_name)
            self.client = session.client('dynamodb', region_name=region_name)
        else:
            self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
            self.client = boto3.client('dynamodb', region_name=region_name)

        self.users_table_name = 'RAG_Users'
        self.conversations_table_name = 'RAG_Conversations'
        self.users_table = None
        self.conversations_table = None

    def create_tables(self):
        """Create DynamoDB tables for users and conversations"""
        try:
            self.users_table = self.dynamodb.create_table(
                TableName=self.users_table_name,
                KeySchema=[{'AttributeName': 'UserID', 'KeyType': 'HASH'}],
                AttributeDefinitions=[{'AttributeName': 'UserID', 'AttributeType': 'S'}],
                BillingMode='PAY_PER_REQUEST'
            )

            self.conversations_table = self.dynamodb.create_table(
                TableName=self.conversations_table_name,
                KeySchema=[
                    {'AttributeName': 'SessionID', 'KeyType': 'HASH'},
                    {'AttributeName': 'MessageID', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'SessionID', 'AttributeType': 'S'},
                    {'AttributeName': 'MessageID', 'AttributeType': 'S'},
                    {'AttributeName': 'UserID', 'AttributeType': 'S'},
                    {'AttributeName': 'Timestamp', 'AttributeType': 'S'}
                ],
                GlobalSecondaryIndexes=[{
                    'IndexName': 'UserID-Timestamp-index',
                    'KeySchema': [
                        {'AttributeName': 'UserID', 'KeyType': 'HASH'},
                        {'AttributeName': 'Timestamp', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }],
                BillingMode='PAY_PER_REQUEST'
            )

            self.users_table.wait_until_exists()
            self.conversations_table.wait_until_exists()
            logger.info("Successfully created DynamoDB tables")

        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                logger.info("Tables already exist, connecting to existing tables")
                self.users_table = self.dynamodb.Table(self.users_table_name)
                self.conversations_table = self.dynamodb.Table(self.conversations_table_name)
            else:
                logger.error(f"Error creating tables: {e}")
                raise

    def connect_to_tables(self):
        """Connect to existing tables"""
        try:
            self.users_table = self.dynamodb.Table(self.users_table_name)
            self.conversations_table = self.dynamodb.Table(self.conversations_table_name)

            self.users_table.load()
            self.conversations_table.load()

            logger.info("Connected to existing DynamoDB tables")

        except ClientError as e:
            logger.error(f"Error connecting to tables: {e}")
            raise

    def create_user(self, user_id: str, name: str, phone_number: str, email: str, 
                   additional_info: str = "") -> Dict[str, Any]:
        """Create a new user"""
        if not self.users_table:
            raise Exception("Users table not initialized")

        user_data = {
            'UserID': user_id,
            'Name': name,
            'PhoneNumber': phone_number,
            'Email': email,
            'AdditionalInfo': additional_info,
            'CreatedAt': datetime.now().isoformat(),
            'UpdatedAt': datetime.now().isoformat()
        }

        try:
            self.users_table.put_item(Item=user_data)
            logger.info(f"Created user: {user_id}")
            return user_data

        except ClientError as e:
            logger.error(f"Error creating user {user_id}: {e}")
            raise

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        if not self.users_table:
            raise Exception("Users table not initialized")

        try:
            response = self.users_table.get_item(Key={'UserID': user_id})
            return response.get('Item')

        except ClientError as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None

    def create_message(self, session_id: str, user_id: str, role: str, 
                      message_content: str, summary_flag: bool = False) -> Dict[str, Any]:
        """Create a new message in conversation"""
        if not self.conversations_table:
            raise Exception("Conversations table not initialized")

        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        message_data = {
            'SessionID': session_id,
            'MessageID': message_id,
            'UserID': user_id,
            'Role': role,
            'MessageContent': message_content,
            'Timestamp': timestamp,
            'SummaryFlag': summary_flag
        }

        try:
            self.conversations_table.put_item(Item=message_data)
            logger.info(f"Created message in session {session_id}")
            return message_data

        except ClientError as e:
            logger.error(f"Error creating message: {e}")
            raise

    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages for a session, ordered by timestamp"""
        if not self.conversations_table:
            raise Exception("Conversations table not initialized")

        try:
            response = self.conversations_table.query(
                KeyConditionExpression='SessionID = :session_id',
                ExpressionAttributeValues={':session_id': session_id},
                ScanIndexForward=True,
                Limit=limit
            )

            messages = response['Items']
            messages.sort(key=lambda x: x['Timestamp'])

            return messages

        except ClientError as e:
            logger.error(f"Error getting session messages: {e}")
            return []

    def get_user_sessions(self, user_id: str, limit: int = 20) -> List[str]:
        """Get all session IDs for a user"""
        if not self.conversations_table:
            raise Exception("Conversations table not initialized")

        try:
            response = self.conversations_table.query(
                IndexName='UserID-Timestamp-index',
                KeyConditionExpression='UserID = :user_id',
                ExpressionAttributeValues={':user_id': user_id},
                ScanIndexForward=False,
                Limit=limit * 10
            )

            session_ids = list(set([item['SessionID'] for item in response['Items']]))
            return session_ids[:limit]

        except ClientError as e:
            logger.error(f"Error getting user sessions: {e}")
            return []

    def get_conversation_context(self, session_id: str, max_turns: int = 10) -> Dict[str, Any]:
        """Get conversation context for RAG, implementing 10-turn limit with summarization"""
        messages = self.get_session_messages(session_id)

        summaries = [msg for msg in messages if msg.get('SummaryFlag', False)]
        regular_messages = [msg for msg in messages if not msg.get('SummaryFlag', False)]

        latest_summary = summaries[-1]['MessageContent'] if summaries else ""

        recent_messages = regular_messages[-max_turns * 2:] if len(regular_messages) > max_turns * 2 else regular_messages

        return {
            'session_id': session_id,
            'summary': latest_summary,
            'recent_messages': recent_messages,
            'total_messages': len(regular_messages),
            'needs_summarization': len(regular_messages) > max_turns * 2
        }

    def create_conversation_summary(self, session_id: str, user_id: str, 
                                  summary_content: str) -> Dict[str, Any]:
        """Create a summary of conversation history"""
        return self.create_message(
            session_id=session_id,
            user_id=user_id,
            role='Summary',
            message_content=summary_content,
            summary_flag=True
        )

    def delete_session(self, session_id: str):
        """Delete all messages in a session"""
        if not self.conversations_table:
            raise Exception("Conversations table not initialized")

        try:
            messages = self.get_session_messages(session_id, limit=1000)

            with self.conversations_table.batch_writer() as batch:
                for message in messages:
                    batch.delete_item(
                        Key={
                            'SessionID': message['SessionID'],
                            'MessageID': message['MessageID']
                        }
                    )

            logger.info(f"Deleted session: {session_id}")

        except ClientError as e:
            logger.error(f"Error deleting session: {e}")
            raise

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                'users_count': 0,
                'sessions_count': 0,
                'messages_count': 0
            }

            if self.users_table:
                users_response = self.users_table.scan(Select='COUNT')
                stats['users_count'] = users_response['Count']

            if self.conversations_table:
                conversations_response = self.conversations_table.scan(Select='COUNT')
                stats['messages_count'] = conversations_response['Count']

                scan_response = self.conversations_table.scan(ProjectionExpression='SessionID')
                session_ids = set([item['SessionID'] for item in scan_response['Items']])
                stats['sessions_count'] = len(session_ids)

            return stats

        except ClientError as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
