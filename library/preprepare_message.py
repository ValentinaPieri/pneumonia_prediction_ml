from .request_message import RequestMessage
from .message import Message
from .message_type import MessageType

class PrePrepareMessage(Message):
    def __init__(self, primary_replica_id: int, view: int, sequence_number: int, request_msg: RequestMessage):
        """
        Initialize a Pre-Prepare message.

        This constructor initializes a message of type `PRE_PREPARE` with a given sequence number, hash chain digest and replica ID.

        Args:
            primary_replica_id (int): The ID of the primary replica.
            view (int): The view number.
            sequence_number (int): The sequence number assigned by the primary replica.
            request_msg (RequestMessage): The request message.

        Attributes:
            primary_replica_id (int): The ID of the primary replica.
            request_msg (RequestMessage): The request message.
        """
        super().__init__(MessageType.PRE_PREPARE, view, sequence_number)
        self.primary_replica_id = primary_replica_id
        self.request_msg = request_msg

    def get_primary_replica_id(self) -> int:
        """
        Get the ID of the primary replica.

        Returns:
            int: The ID of the primary replica.
        """
        return self.primary_replica_id

    def get_view(self) -> int:
        """
        Get the view number.

        Returns:
            int: The view number.
        """
        return self.view

    def get_sequence_number(self) -> int:
        """
        Get the sequence number assigned by the primary replica.

        Returns:
            int: The sequence number assigned by the primary replica.
        """
        return self.sequence_number

    def get_request_msg(self):
        """
        Get the request message.

        Returns:
            RequestMessage: The request message.
        """
        return self.request_msg

    def to_string(self):
        """
        Convert the message to a string.

        Returns:
            str: The string representation of the message.
        """
        return super().to_string() + f"Primary Replica ID: {self.primary_replica_id}, Hashed Request Message: {self.request_msg}"