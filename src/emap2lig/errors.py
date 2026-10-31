"""Exceptions raised by the Emap2lig pipeline."""


class Emap2ligError(RuntimeError):
    """Base class for pipeline failures the user can act on."""


class NoBlobsDetectedError(Emap2ligError):
    """The find stage produced no ligand density blobs."""


class TooManyBlobsError(Emap2ligError):
    """The find stage produced more blobs than are worth modeling."""


class LigandPreparationError(Emap2ligError):
    """A ligand record could not be turned into a :class:`LigandObject`."""


class JobCancelledError(Emap2ligError):
    """A web job was cancelled between pipeline progress events."""
