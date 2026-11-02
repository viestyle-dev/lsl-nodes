import logging
from nonoha_core.node_registry import NodeRegistry
from .lsl_nodes import (
    LslOutletNode,
    LslInletNode,
    LslStreamResolverNode,
)

logger = logging.getLogger(__name__)


def register_nodes(registry: NodeRegistry):
    logger.info("lsl-nodes plugin loaded")

    registry.register_node("LslOutlet", LslOutletNode)
    registry.register_node("LslInlet", LslInletNode)
    registry.register_node("LslStreamResolver", LslStreamResolverNode)


def finalize():
    logger.info("lsl-nodes plugin finalized")
