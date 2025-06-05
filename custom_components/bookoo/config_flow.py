"""Config flow for Bookoo integration."""

import logging
from typing import Any

import voluptuous as vol
from aiobookoov2.exceptions import (
    BookooDeviceNotFound,
    BookooError,
    BookooUnknownDevice,
)
from aiobookoov2.helpers import is_bookoo_scale
from homeassistant.components.bluetooth import (
    BluetoothServiceInfoBleak,
    async_discovered_service_info,
)
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_ADDRESS, CONF_NAME
from homeassistant.core import callback  # Added for @callback decorator
from homeassistant.helpers.device_registry import format_mac
from homeassistant.helpers.selector import (
    BooleanSelector,  # Added
    BooleanSelectorConfig,  # Added
    EntitySelector,
    EntitySelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    CONF_IS_VALID_SCALE,
    DEFAULT_AUTO_STOP_FLOW_CUTOFF_THRESHOLD,
    DEFAULT_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY,
    DEFAULT_AUTO_STOP_MIN_DURATION_FOR_CUTOFF,
    DEFAULT_AUTO_STOP_MIN_DURATION_FOR_STABILITY,
    DEFAULT_AUTO_STOP_MIN_FLOW_FOR_STABILITY,
    DEFAULT_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION,
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_CONNECT_TIMEOUT,
    DOMAIN,
    OPTION_AUTO_STOP_FLOW_CUTOFF_THRESHOLD,
    OPTION_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY,
    OPTION_AUTO_STOP_MIN_DURATION_FOR_CUTOFF,
    OPTION_AUTO_STOP_MIN_DURATION_FOR_STABILITY,
    OPTION_AUTO_STOP_MIN_FLOW_FOR_STABILITY,
    OPTION_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION,
    OPTION_COMMAND_TIMEOUT,
    OPTION_CONNECT_TIMEOUT,
    OPTION_ENABLE_AUTO_STOP_FLOW_CUTOFF,
    OPTION_LINKED_BEAN_WEIGHT_ENTITY,
    OPTION_LINKED_BREW_TEMPERATURE_ENTITY,
    OPTION_LINKED_COFFEE_NAME_ENTITY,
    OPTION_LINKED_GRIND_SETTING_ENTITY,
    OPTION_MIN_SHOT_DURATION,
)

_LOGGER = logging.getLogger(__name__)


class BookooConfigFlow(ConfigFlow, domain=DOMAIN):  # type: ignore[call-arg]
    """Handle a config flow for bookoo."""

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._discovered: dict[str, Any] = {}
        self._discovered_devices: dict[str, str] = {}  # Keep this for user step

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> "BookooOptionsFlowHandler":
        """Get the options flow for this handler."""
        return BookooOptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step when a user initiates a config flow.

        This step allows the user to select a discovered Bluetooth device or
        initiate a new scan if no devices were automatically found.
        """
        """Handle a flow initialized by the user."""

        errors: dict[str, str] = {}

        if user_input is not None:
            mac = user_input[CONF_ADDRESS]
            try:
                is_valid_bookoo_scale = await is_bookoo_scale(mac)
            except BookooDeviceNotFound:
                errors["base"] = "device_not_found"
            except BookooError:
                _LOGGER.exception("Error occurred while connecting to the scale")
                errors["base"] = "unknown"
            except BookooUnknownDevice:
                return self.async_abort(reason="unsupported_device")
            else:
                await self.async_set_unique_id(format_mac(mac))
                self._abort_if_unique_id_configured()

            if not errors:
                return self.async_create_entry(
                    title=self._discovered_devices[mac],
                    data={
                        CONF_ADDRESS: mac,
                        CONF_IS_VALID_SCALE: is_valid_bookoo_scale,
                    },
                )

        for device in async_discovered_service_info(self.hass):
            self._discovered_devices[device.address] = device.name

        if not self._discovered_devices:
            return self.async_abort(reason="no_devices_found")

        options = [
            SelectOptionDict(
                value=device_mac,
                label=f"{device_name} ({device_mac})",
            )
            for device_mac, device_name in self._discovered_devices.items()
        ]

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_ADDRESS): SelectSelector(
                        SelectSelectorConfig(
                            options=options,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    )
                }
            ),
            errors=errors,
        )

    async def async_step_bluetooth(
        self, discovery_info: BluetoothServiceInfoBleak
    ) -> ConfigFlowResult:
        """Handle a flow initiated by Bluetooth discovery.

        Stores the discovered device information and proceeds to confirmation.
        It also checks if the device is a valid Bookoo scale.
        """
        """Handle a discovered Bluetooth device."""

        self._discovered[CONF_ADDRESS] = discovery_info.address
        self._discovered[CONF_NAME] = discovery_info.name

        await self.async_set_unique_id(format_mac(discovery_info.address))
        self._abort_if_unique_id_configured()

        try:
            self._discovered[CONF_IS_VALID_SCALE] = await is_bookoo_scale(
                discovery_info.address
            )
        except BookooDeviceNotFound:
            _LOGGER.debug("Device not found during discovery")
            return self.async_abort(reason="device_not_found")
        except BookooError:
            _LOGGER.debug(
                "Error occurred while connecting to the scale during discovery",
                exc_info=True,
            )
            return self.async_abort(reason="unknown")
        except BookooUnknownDevice:
            _LOGGER.debug("Unsupported device during discovery")
            return self.async_abort(reason="unsupported_device")

        return await self.async_step_bluetooth_confirm()

    async def async_step_bluetooth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the step to confirm a discovered Bluetooth device.

        Allows the user to confirm adding the discovered scale and optionally
        set a custom name for it in Home Assistant.
        """

        if user_input is not None:
            # User has submitted the confirmation form (which now includes the name)
            # Use the name from user_input if provided, otherwise default to discovered name
            title = user_input.get(CONF_NAME, self._discovered[CONF_NAME])
            return self.async_create_entry(
                title=title,
                data={
                    CONF_ADDRESS: self._discovered[CONF_ADDRESS],
                    CONF_IS_VALID_SCALE: self._discovered[CONF_IS_VALID_SCALE],
                    # Optionally store the user-provided name in data if needed elsewhere,
                    # but title is the primary use.
                    # CONF_NAME: title
                },
            )

        # Show the confirmation form to the user
        # Pre-fill the name field with the discovered Bluetooth name
        current_name = self._discovered[CONF_NAME]
        self.context["title_placeholders"] = {
            CONF_NAME: current_name
        }  # For description placeholders if any

        return self.async_show_form(
            step_id="bluetooth_confirm",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_NAME, default=current_name): str,
                }
            ),
            description_placeholders={
                CONF_NAME: current_name
            },  # Pass current_name for use in translation string
            # self._set_confirm_only() is usually for forms with no user input fields, just a confirm button.
            # Since we added a name field, it's a regular form step now.
        )


class BookooOptionsFlowHandler(OptionsFlow):
    """Handle an options flow for Bookoo."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry  # Store config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the Bookoo integration options.

        Allows users to configure settings such as linked entities for shot parameters,
        auto-stop features, and Bluetooth connection timeouts.
        """
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validation is now primarily handled by voluptuous schema
            # If specific cross-field validation is needed, it can be added here.
            return self.async_create_entry(title="", data=user_input)

        # Get current options to pre-fill the form
        current_options = self.config_entry.options

        options_schema = vol.Schema(
            {
                vol.Optional(
                    OPTION_MIN_SHOT_DURATION,
                    default=current_options.get(
                        OPTION_MIN_SHOT_DURATION, 10
                    ),  # Default to 10s
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=300.0)),
                vol.Optional(
                    OPTION_LINKED_BEAN_WEIGHT_ENTITY,
                    default=current_options.get(
                        OPTION_LINKED_BEAN_WEIGHT_ENTITY, vol.UNDEFINED
                    ),
                ): EntitySelector(
                    EntitySelectorConfig(domain="input_number", multiple=False)
                ),
                vol.Optional(
                    OPTION_LINKED_COFFEE_NAME_ENTITY,
                    default=current_options.get(
                        OPTION_LINKED_COFFEE_NAME_ENTITY, vol.UNDEFINED
                    ),
                ): EntitySelector(
                    EntitySelectorConfig(domain="input_text", multiple=False)
                ),
                vol.Optional(
                    OPTION_LINKED_GRIND_SETTING_ENTITY,
                    default=current_options.get(
                        OPTION_LINKED_GRIND_SETTING_ENTITY, vol.UNDEFINED
                    ),
                ): EntitySelector(
                    EntitySelectorConfig(
                        domain="input_text", multiple=False
                    )  # Assuming text for flexibility
                ),
                vol.Optional(
                    OPTION_LINKED_BREW_TEMPERATURE_ENTITY,
                    default=current_options.get(
                        OPTION_LINKED_BREW_TEMPERATURE_ENTITY, vol.UNDEFINED
                    ),
                ): EntitySelector(
                    EntitySelectorConfig(
                        domain="input_number", multiple=False
                    )  # Assuming numeric input
                ),
                # Auto-Stop Options
                vol.Optional(
                    OPTION_ENABLE_AUTO_STOP_FLOW_CUTOFF,
                    default=current_options.get(
                        OPTION_ENABLE_AUTO_STOP_FLOW_CUTOFF, False
                    ),
                ): BooleanSelector(BooleanSelectorConfig()),
                vol.Optional(
                    OPTION_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION,
                    default=current_options.get(
                        OPTION_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION,
                        DEFAULT_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION,
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=60.0)),
                vol.Optional(
                    OPTION_AUTO_STOP_MIN_FLOW_FOR_STABILITY,
                    default=current_options.get(
                        OPTION_AUTO_STOP_MIN_FLOW_FOR_STABILITY,
                        DEFAULT_AUTO_STOP_MIN_FLOW_FOR_STABILITY,
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=10.0)),
                vol.Optional(
                    OPTION_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY,
                    default=current_options.get(
                        OPTION_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY,
                        DEFAULT_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY,
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=100.0)),
                vol.Optional(
                    OPTION_AUTO_STOP_MIN_DURATION_FOR_STABILITY,
                    default=current_options.get(
                        OPTION_AUTO_STOP_MIN_DURATION_FOR_STABILITY,
                        DEFAULT_AUTO_STOP_MIN_DURATION_FOR_STABILITY,
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=60.0)),
                vol.Optional(
                    OPTION_AUTO_STOP_FLOW_CUTOFF_THRESHOLD,
                    default=current_options.get(
                        OPTION_AUTO_STOP_FLOW_CUTOFF_THRESHOLD,
                        DEFAULT_AUTO_STOP_FLOW_CUTOFF_THRESHOLD,
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=5.0)),
                vol.Optional(
                    OPTION_AUTO_STOP_MIN_DURATION_FOR_CUTOFF,
                    default=current_options.get(
                        OPTION_AUTO_STOP_MIN_DURATION_FOR_CUTOFF,
                        DEFAULT_AUTO_STOP_MIN_DURATION_FOR_CUTOFF,
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=60.0)),
                # Bluetooth Timeout Options
                vol.Optional(
                    OPTION_CONNECT_TIMEOUT,
                    default=current_options.get(
                        OPTION_CONNECT_TIMEOUT, DEFAULT_CONNECT_TIMEOUT
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=120.0)),
                vol.Optional(
                    OPTION_COMMAND_TIMEOUT,
                    default=current_options.get(
                        OPTION_COMMAND_TIMEOUT, DEFAULT_COMMAND_TIMEOUT
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=60.0)),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,  # Voluptuous will populate errors if validation fails on submit
            # Description placeholders can be added here if desired, similar to how they were before.
            # For brevity, they are omitted here but can be added back for better UX.
        )
