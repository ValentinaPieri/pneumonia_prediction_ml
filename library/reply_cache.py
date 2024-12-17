from .reply_message import ReplyMessage

class ReplyCache:
    """
    The ReplyCache is responsible for storing the last reply sent to each client to prevent re-executing
    the same operations and ensure idempotency in the protocol. This cache allows replicas to quickly
    respond to repeated client requests by returning the cached reply instead of processing the operation again.
    """
    def __init__(self):
        """
        Initializes an empty ReplyCache.

        The ReplyCache starts as an empty dictionary, mapping client IDs to their most recent replies.
        """
        self.cache = {}

    def update_cache(self, client_id, reply : ReplyMessage):
        """
        Update the cache of a client with a reply message.

        Args:
            client_id (int): The client ID.
            reply (ReplyMessage): The reply message.
        """
        self.cache[client_id] = reply

    def get_reply(self, client_id) -> ReplyMessage:
        """
        Retrieves the most recent reply for a client.

        Args:
            client_id (int): The client ID.

        Returns:
            ReplyMessage: The reply message.
        """
        return self.cache[client_id] if client_id in self.cache else None

    def clear(self):
        """
        Clears all entries from the reply cache.

        This method is typically called when the replica needs to reset or garbage collect old entries.
        """
        self.cache.clear()