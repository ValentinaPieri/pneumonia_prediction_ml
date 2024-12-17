from .message import Message
from .message_type import MessageType

class OperationsDict(Message):
    def __init__(self, dict: dict, view: int = None, sequence_number: int = None):
        """
        Initialize an Operations Dictionary message.

        This constructor initializes a message of type `OPERATIONS_DICTIONARY` with a given dictionary of operations. The dictionary is used to store the operations that are in progress during a view change.

        Args:
            dict (dict): A dictionary of operations.
            view (int): The view number.
            sequence_number (int): The sequence number assigned by the primary replica.

        Attributes:
            dict (dict): A dictionary of operations.
        """
        super().__init__(MessageType.OPERATIONS_DICTIONARY, view, sequence_number)
        self.dict = dict

    def get_dict(self) -> dict:
        """
        Get the dictionary of operations.

        Returns:
            dict: A dictionary of operations.
        """
        return self.dict