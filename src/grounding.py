"""Citation tokens shared by the reply and the prompt builder.

Kept in its own module so the prompt builder can label a fact or a chunk
without importing the session, which imports the prompt builder.
"""

from __future__ import annotations


def chunk_cite(hit: dict) -> str:
    """The citation for one retrieved chunk: [doc_id#chunk_id].

    chunk_id already names the document, the version and the piece. The
    doc_id is repeated in front so the form matches [doc_id#chunk].
    """
    return f"[{hit.get('doc_id')}#{hit.get('chunk_id')}]"


def memory_cite(memory_id: str | None) -> str:
    """The citation for one fact or episode: [mem:id]."""
    return f"[mem:{memory_id}]"
