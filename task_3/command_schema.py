"""JSON schema for Task 3 robot commands."""


COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "accepted": {
            "type": "boolean"
        },

        "reason": {
            "type": ["string", "null"]
        },

        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "move",
                            "turn",
                            "goto_object",
                            "stop",
                            "chat",
                        ],
                    },

                    "vx": {
                        "type": ["number", "null"]
                    },

                    "vy": {
                        "type": ["number", "null"]
                    },

                    "wz": {
                        "type": ["number", "null"]
                    },

                    "duration": {
                        "type": ["number", "null"]
                    },

                    "angle_deg": {
                        "type": ["number", "null"]
                    },

                    "class": {
                        "type": ["string", "null"]
                    },

                    "color": {
                        "type": ["string", "null"]
                    },

                    "reply": {
                        "type": ["string", "null"]
                    },
                },

                "required": [
                    "action",
                    "vx",
                    "vy",
                    "wz",
                    "duration",
                    "angle_deg",
                    "class",
                    "color",
                    "reply",
                ],

                "additionalProperties": False,
            },
        },
    },

    "required": [
        "accepted",
        "reason",
        "actions",
    ],

    "additionalProperties": False,
}