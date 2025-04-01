import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainETL:
    """
    Extract, Transform, Load pipeline for blockchain transaction data
    Handles large-scale data processing with parallel execution
    """
    
    def __init__(self, node_url: str = "http://localhost:8545", batch_size: int = 1000):
        self.node_url = node_url
        self.batch_size = batch_size
        self.contract_abis = {}  # Cache for contract ABIs
    
    def extract_blocks(self, start_block: int, end_block: int) -> List[Dict]:
        """Extract block data from blockchain node"""
        blocks = []
        for block_num in range(start_block, end_block + 1):
            try:
                block_data = self._rpc_call("eth_getBlockByNumber", [hex(block_num), True])
                if block_data:
                    blocks.append(self._transform_block(block_data))
            except Exception as e:
                logger.error(f"Error extracting block {block_num}: {str(e)}")
                continue
        return blocks
    
    def extract_transactions(self, start_block: int, end_block: int) -> List[Dict]:
        """Extract transactions from blocks"""
        transactions = []
        blocks = self.extract_blocks(start_block, end_block)
        
        for block in blocks:
            for tx in block.get('transactions', []):
                tx['block_number'] = block['number']
                tx['block_timestamp'] = block['timestamp']
                transactions.append(tx)
        
        return transactions
    
    def extract_contract_data(self, contract_address: str, abi: Dict, 
                            start_block: int, end_block: int) -> List[Dict]:
        """Extract contract events and state changes"""
        self.contract_abis[contract_address] = abi
        events = []
        
        # Get all events in block range
        event_signatures = self._get_event_signatures(abi)
        for signature in event_signatures:
            events.extend(self._get_events(contract_address, signature, start_block, end_block))
        
        return events
    
    def parallel_extract(self, start_block: int, end_block: int, 
                        workers: int = 4) -> List[Dict]:
        """Parallel block extraction"""
        block_ranges = self._create_ranges(start_block, end_block, workers)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self.extract_blocks, 
                                      [r[0] for r in block_ranges],
                                      [r[1] for r in block_ranges]))
        
        # Flatten results
        return [block for sublist in results for block in sublist]
    
    def transform_to_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Convert raw blockchain data to pandas DataFrame"""
        return pd.DataFrame(data)
    
    def load_to_parquet(self, df: pd.DataFrame, output_path: str, 
                       partition_cols: Optional[List[str]] = None) -> None:
        """Save DataFrame to Parquet format"""
        table = pa.Table.from_pandas(df)
        
        if partition_cols:
            pq.write_to_dataset(table, 
                              root_path=output_path,
                              partition_cols=partition_cols)
        else:
            pq.write_table(table, output_path)
    
    def run_pipeline(self, start_block: int, end_block: int, 
                    output_path: str, workers: int = 4) -> None:
        """Complete ETL pipeline"""
        logger.info(f"Starting ETL pipeline for blocks {start_block} to {end_block}")
        
        # Extract
        blocks = self.parallel_extract(start_block, end_block, workers)
        transactions = [tx for block in blocks for tx in block.get('transactions', [])]
        
        # Transform
        blocks_df = self.transform_to_dataframe(blocks)
        transactions_df = self.transform_to_dataframe(transactions)
        
        # Add date partitions
        blocks_df['date'] = pd.to_datetime(blocks_df['timestamp'], unit='s').dt.date
        transactions_df['date'] = pd.to_datetime(transactions_df['block_timestamp'], unit='s').dt.date
        
        # Load
        os.makedirs(output_path, exist_ok=True)
        self.load_to_parquet(blocks_df, os.path.join(output_path, 'blocks'))
        self.load_to_parquet(transactions_df, os.path.join(output_path, 'transactions'))
        
        logger.info(f"ETL pipeline completed. Data saved to {output_path}")
    
    def _rpc_call(self, method: str, params: List) -> Optional[Dict]:
        """Make JSON-RPC call to blockchain node"""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        try:
            response = requests.post(self.node_url, json=payload)
            return response.json().get('result')
        except Exception as e:
            logger.error(f"RPC call failed: {str(e)}")
            return None
    
    def _transform_block(self, block: Dict) -> Dict:
        """Transform raw block data"""
        return {
            'number': int(block['number'], 16),
            'hash': block['hash'],
            'parent_hash': block['parentHash'],
            'nonce': block['nonce'],
            'sha3_uncles': block['sha3Uncles'],
            'logs_bloom': block['logsBloom'],
            'transactions_root': block['transactionsRoot'],
            'state_root': block['stateRoot'],
            'receipts_root': block['receiptsRoot'],
            'miner': block['miner'],
            'difficulty': int(block['difficulty'], 16),
            'total_difficulty': int(block['totalDifficulty'], 16),
            'size': int(block['size'], 16),
            'extra_data': block['extraData'],
            'gas_limit': int(block['gasLimit'], 16),
            'gas_used': int(block['gasUsed'], 16),
            'timestamp': int(block['timestamp'], 16),
            'transaction_count': len(block['transactions']),
            'transactions': [self._transform_transaction(tx) for tx in block['transactions']]
        }
    
    def _transform_transaction(self, tx: Dict) -> Dict:
        """Transform raw transaction data"""
        return {
            'hash': tx['hash'],
            'nonce': int(tx['nonce'], 16),
            'block_hash': tx['blockHash'],
            'block_number': int(tx['blockNumber'], 16),
            'transaction_index': int(tx['transactionIndex'], 16),
            'from': tx['from'],
            'to': tx['to'],
            'value': int(tx['value'], 16),
            'gas': int(tx['gas'], 16),
            'gas_price': int(tx['gasPrice'], 16),
            'input': tx['input'],
            'v': tx['v'],
            'r': tx['r'],
            's': tx['s']
        }
    
    def _get_event_signatures(self, abi: Dict) -> List[str]:
        """Get all event signatures from contract ABI"""
        return [
            entry['name'] for entry in abi 
            if entry['type'] == 'event'
        ]
    
    def _get_events(self, contract_address: str, event_sig: str, 
                   start_block: int, end_block: int) -> List[Dict]:
        """Get events from blockchain"""
        topic = self._get_event_topic(event_sig)
        params = {
            "fromBlock": hex(start_block),
            "toBlock": hex(end_block),
            "address": contract_address,
            "topics": [topic]
        }
        
        logs = self._rpc_call("eth_getLogs", [params])
        if not logs:
            return []
        
        return [self._transform_log(log) for log in logs]
    
    def _get_event_topic(self, event_sig: str) -> str:
        """Get topic hash for event signature"""
        return '0x' + hashlib.sha3_256(event_sig.encode()).hexdigest()
    
    def _transform_log(self, log: Dict) -> Dict:
        """Transform raw log/event data"""
        return {
            'log_index': int(log['logIndex'], 16),
            'transaction_index': int(log['transactionIndex'], 16),
            'transaction_hash': log['transactionHash'],
            'block_hash': log['blockHash'],
            'block_number': int(log['blockNumber'], 16),
            'address': log['address'],
            'data': log['data'],
            'topics': log['topics'],
            'removed': log['removed']
        }
    
    def _create_ranges(self, start: int, end: int, chunks: int) -> List[Tuple[int, int]]:
        """Create block ranges for parallel processing"""
        total_blocks = end - start + 1
        chunk_size = total_blocks // chunks
        ranges = []
        
        for i in range(chunks):
            chunk_start = start + i * chunk_size
            chunk_end = chunk_start + chunk_size - 1 if i < chunks - 1 else end
            ranges.append((chunk_start, chunk_end))
        
        return ranges
