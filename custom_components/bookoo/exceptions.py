"""Custom exceptions for the Bookoo integration."""


class BookooIntegrationError(Exception):
    """Base class for exceptions raised by the Bookoo integration."""

    pass


class BookooScaleConnectionError(BookooIntegrationError):
    """Raised when there's an error connecting to the Bookoo scale."""

    pass


class BookooScaleCommandError(BookooIntegrationError):
    """Raised when there's an error sending a command to the Bookoo scale."""

    pass
