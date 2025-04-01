import json
from dataclasses import dataclass
from typing import Dict, Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

@dataclass
class Wallet:
    private_key: ec.EllipticCurvePrivateKey
    public_key: ec.EllipticCurvePublicKey

    @classmethod
    def generate(cls) -> 'Wallet':
        """Generate a new wallet with private and public keys"""
        private_key = ec.generate_private_key(ec.SECP256K1())
        public_key = private_key.public_key()
        return cls(private_key, public_key)

    def get_address(self) -> str:
        """Get the wallet address from public key"""
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(public_bytes).hexdigest()

    def sign(self, data: Dict[str, Any]) -> str:
        """Sign data with private key"""
        data_string = json.dumps(data, sort_keys=True).encode()
        signature = self.private_key.sign(
            data_string,
            ec.ECDSA(hashes.SHA256())
        return signature.hex()

    @staticmethod
    def verify(public_key: ec.EllipticCurvePublicKey, data: Dict[str, Any], signature: str) -> bool:
        """Verify signature with public key"""
        data_string = json.dumps(data, sort_keys=True).encode()
        try:
            public_key.verify(
                bytes.fromhex(signature),
                data_string,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except InvalidSignature:
            return False

    def serialize(self) -> Dict[str, str]:
        """Serialize wallet to dict"""
        private_bytes = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return {
            'private_key': private_bytes.decode(),
            'public_key': public_bytes.decode(),
            'address': self.get_address()
        }

    @classmethod
    def deserialize(cls, data: Dict[str, str]) -> 'Wallet':
        """Deserialize wallet from dict"""
        private_key = serialization.load_pem_private_key(
            data['private_key'].encode(),
            password=None
        )
        public_key = serialization.load_pem_public_key(
            data['public_key'].encode()
        )
        return cls(private_key, public_key)
