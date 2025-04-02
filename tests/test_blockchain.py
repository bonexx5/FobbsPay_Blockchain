import pytest
from core.blockchain import Blockchain, Block
from core.wallet import Wallet
import time

@pytest.fixture
def blockchain():
    return Blockchain(node_id="test_node")

@pytest.fixture
def wallet():
    return Wallet.generate()

def test_create_genesis_block(blockchain):
    assert len(blockchain.chain) == 1
    genesis = blockchain.chain[0]
    assert genesis.index == 1
    assert genesis.previous_hash == "1"

def test_new_transaction(blockchain, wallet):
    sender = wallet.get_address()
    recipient = Wallet.generate().get_address()
    
    index = blockchain.new_transaction(sender, recipient, 1.0)
    assert index == 2  # Will be added to next block (after genesis)
    assert len(blockchain.current_transactions) == 1

def test_mine_block(blockchain, wallet):
    # Add a transaction
    sender = wallet.get_address()
    recipient = Wallet.generate().get_address()
    blockchain.new_transaction(sender, recipient, 1.0)
    
    # Mine block
    last_block = blockchain.last_block
    last_proof = last_block.proof
    proof = blockchain.proof_of_work(last_proof)
    
    block = blockchain.new_block(proof)
    
    assert block.index == 2
    assert len(block.transactions) == 1
    assert block.previous_hash == blockchain.hash(last_block)
    assert blockchain.valid_proof(last_proof, proof)

def test_chain_validation(blockchain, wallet):
    # Create a valid chain
    sender = wallet.get_address()
    recipient = Wallet.generate().get_address()
    
    blockchain.new_transaction(sender, recipient, 1.0)
    last_proof = blockchain.last_block.proof
    proof = blockchain.proof_of_work(last_proof)
    blockchain.new_block(proof)
    
    # Test validation
    assert blockchain.valid_chain(blockchain.chain)
    
    # Tamper with chain
    blockchain.chain[1].transactions[0]['amount'] = 100.0
    assert not blockchain.valid_chain(blockchain.chain)

def test_get_balance(blockchain, wallet):
    sender = wallet.get_address()
    recipient = Wallet.generate().get_address()
    
    blockchain.new_transaction(sender, recipient, 1.0)
    blockchain.new_transaction("0", sender, 1.0)  # Mining reward
    
    last_proof = blockchain.last_block.proof
    proof = blockchain.proof_of_work(last_proof)
    blockchain.new_block(proof)
    
    assert blockchain.get_balance(sender) == 0.0  # 1 received - 1 sent
    assert blockchain.get_balance(recipient) == 1.0
