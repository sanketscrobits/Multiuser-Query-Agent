from contextvars import ContextVar

namespace_context: ContextVar[str] = ContextVar("namespace_context", default=None)

def set_namespace(namespace: str):
    namespace_context.set(namespace)

def get_namespace() -> str:
    return namespace_context.get()
