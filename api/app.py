from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import uvicorn
import jwt
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path

from core.blockchain import Blockchain
from core.wallet import Wallet
from data_pipeline.optimization import DataOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FobbsPay Blockchain API",
    description="REST API for FobbsPay blockchain fintech payment system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "fobbspay-secret-key")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Initialize components
blockchain = Blockchain(node_id="fobbspay_node_1")
data_optimizer = DataOptimizer(config_path="config/data_config.json")

# Models
class TransactionRequest(BaseModel):
    sender: str = Field(..., description="Sender wallet address")
    recipient: str = Field(..., description="Recipient wallet address")
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    metadata: Optional[Dict] = Field(None, description="Optional transaction metadata")
    signature: Optional[str] = Field(None, description="Digital signature of the transaction")

    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return round(v, 8)  # Standardize to 8 decimal places

class BlockResponse(BaseModel):
    index: int
    timestamp: float
    transactions: List[Dict]
    proof: int
    previous_hash: str
    hash: str

class WalletResponse(BaseModel):
    address: str
    public_key: str
    balance: float
    transaction_count: int

class AnalyticsResponse(BaseModel):
    transaction_count: int
    total_amount: float
    avg_amount: float
    active_users: int
    anomaly_count: int
    cluster_distribution: Dict[int, int]

class FraudPredictionRequest(BaseModel):
    transactions: List[Dict]

class FraudPredictionResponse(BaseModel):
    predictions: List[float]
    explanations: List[Dict]

# Helper functions
def create_jwt_token(address: str) -> str:
    """Create JWT token for authenticated requests"""
    expiration = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "address": address,
        "exp": expiration
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Optional[Dict]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Dependency to get current user from JWT"""
    token = credentials.credentials
    payload = verify_jwt_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload["address"]

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# API Endpoints
@app.post("/transactions", response_model=Dict, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    transaction: TransactionRequest,
    current_user: str = Depends(get_current_user)
):
    """Create a new transaction"""
    if current_user != transaction.sender:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only create transactions from your own address"
        )
    
    index = blockchain.new_transaction(
        transaction.sender,
        transaction.recipient,
        transaction.amount,
        transaction.metadata,
        transaction.signature
    )
    
    logger.info(f"New transaction created by {current_user} to {transaction.recipient}")
    return {"message": f"Transaction will be added to Block {index}", "status": "pending"}

@app.get("/chain", response_model=List[BlockResponse])
async def get_blockchain():
    """Return the full blockchain"""
    chain = []
    for block in blockchain.chain:
        block_data = {
            "index": block.index,
            "timestamp": block.timestamp,
            "transactions": block.transactions,
            "proof": block.proof,
            "previous_hash": block.previous_hash,
            "hash": blockchain.hash(block)
        }
        chain.append(block_data)
    return chain

@app.post("/mine", response_model=BlockResponse)
async def mine_block(current_user: str = Depends(get_current_user)):
    """Mine a new block"""
    if not blockchain.current_transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No transactions to mine"
        )
    
    last_block = blockchain.last_block
    last_proof = last_block.proof
    proof = blockchain.proof_of_work(last_proof)
    
    # Reward the miner
    blockchain.new_transaction(
        sender="0",  # Represents the blockchain network
        recipient=current_user,
        amount=1.0,  # Mining reward
        metadata={"type": "mining_reward"}
    )
    
    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(proof, previous_hash)
    
    logger.info(f"New block mined by {current_user}")
    
    return {
        "index": block.index,
        "timestamp": block.timestamp,
        "transactions": block.transactions,
        "proof": block.proof,
        "previous_hash": block.previous_hash,
        "hash": blockchain.hash(block)
    }

@app.get("/wallet/new", response_model=WalletResponse)
async def create_wallet():
    """Generate a new wallet"""
    wallet = Wallet.generate()
    address = wallet.get_address()
    
    return {
        "address": address,
        "public_key": wallet.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode(),
        "balance": 0.0,
        "transaction_count": 0
    }

@app.get("/wallet/{address}", response_model=WalletResponse)
async def get_wallet(address: str):
    """Get wallet information"""
    balance = blockchain.get_balance(address)
    transactions = blockchain.get_transaction_history(address)
    
    return {
        "address": address,
        "public_key": "",  # In real app, this would be fetched from a registry
        "balance": balance,
        "transaction_count": len(transactions)
    }

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get blockchain analytics"""
    transactions = [tx for block in blockchain.chain for tx in block.transactions]
    amounts = [tx["amount"] for tx in transactions if isinstance(tx.get("amount"), (int, float))]
    senders = [tx["sender"] for tx in transactions if tx.get("sender")]
    recipients = [tx["recipient"] for tx in transactions if tx.get("recipient")]
    
    unique_users = set(senders + recipients)
    cluster_dist = {}
    
    # In a real app, we'd use the data pipeline for this
    if hasattr(blockchain, 'cluster_transactions'):
        cluster_dist = blockchain.cluster_transactions(transactions)
    
    return {
        "transaction_count": len(transactions),
        "total_amount": sum(amounts),
        "avg_amount": sum(amounts) / len(amounts) if amounts else 0,
        "active_users": len(unique_users),
        "anomaly_count": 0,  # Would come from anomaly detection
        "cluster_distribution": cluster_dist
    }

@app.post("/predict/fraud", response_model=FraudPredictionResponse)
async def predict_fraud(request: FraudPredictionRequest):
    """Predict fraud probability for transactions"""
    try:
        df = pd.DataFrame(request.transactions)
        predictions = data_optimizer.predict(df)
        
        # Generate explanations if SHAP explainer is available
        explanations = []
        if data_optimizer.shap_explainer:
            processed_data = data_optimizer.transform_data(
                data_optimizer.feature_engineer.transform(
                    data_optimizer.clean_data(df)
                )
            )
            shap_values = data_optimizer.shap_explainer.shap_values(processed_data)
            
            for i in range(len(request.transactions)):
                explanations.append({
                    "features": processed_data.iloc[i].to_dict(),
                    "shap_values": shap_values[i].tolist()
                })
        
        return {
            "predictions": predictions.tolist(),
            "explanations": explanations
        }
    except Exception as e:
        logger.error(f"Fraud prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not process prediction request"
        )

@app.post("/auth/login")
async def login(wallet_data: Dict):
    """Authenticate wallet and get JWT token"""
    # In a real app, this would verify wallet signature
    address = wallet_data.get("address")
    if not address:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Wallet address is required"
        )
    
    token = create_jwt_token(address)
    return {"access_token": token, "token_type": "bearer"}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting FobbsPay Blockchain API")
    
    # Load data optimization models if they exist
    if Path("models").exists():
        data_optimizer.load_models()
        logger.info("Loaded trained data optimization models")
    
    # In a real app, we'd also:
    # - Connect to database
    # - Initialize cache
    # - Start background tasks
    # - etc.

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
