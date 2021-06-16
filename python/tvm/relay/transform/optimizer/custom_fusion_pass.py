from enum import IntEnum

CONFIG_VAR_USER_DEFINED_FUSION_PASS = "relay.FuseOps.UserDefinedFusion"

class CustomFusionPass(IntEnum):
    # This is for measurement
    USER_DEFINED_FUSION = 0
    DP = 1
    EXHAUSTIVE_SEARCH = 2
    TWO_LEVEL_OPT = 3

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_