"""Config flow for Bookoo integration."""

import logging
from typing import Any

from aiobookoov2.exceptions import BookooDeviceNotFound, BookooError, BookooUnknownDevice
from aiobookoov2.helpers import is_bookoo_scale
import voluptuous as vol

from homeassistant.components.bluetooth import (
    BluetoothServiceInfoBleak,
    async_discovered_service_info,
)
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback  # Added for @callback decorator
from homeassistant.const import CONF_ADDRESS, CONF_NAME
from homeassistant.helpers.device_registry import format_mac
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    NumberSelector,  # Added
    NumberSelectorConfig,  # Added
    NumberSelectorMode,  # Added
    EntitySelector,  # Added
    EntitySelectorConfig,  # Added
)

from .const import (
    CONF_IS_VALID_SCALE, 
    DOMAIN,
    OPTION_MIN_SHOT_DURATION,
    OPTION_LINKED_BEAN_WEIGHT_ENTITY,
    OPTION_LINKED_COFFEE_NAME_ENTITY
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
        config_entry: ConfigFlowResult,  # ConfigEntry is more appropriate here
    ) -> "BookooOptionsFlowHandler":
        """Get the options flow for this handler."""
        return BookooOptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
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
        """Handle confirmation of Bluetooth discovery."""

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
        self.context["title_placeholders"] = {CONF_NAME: current_name} # For description placeholders if any

        return self.async_show_form(
            step_id="bluetooth_confirm",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_NAME, default=current_name): str,
                }
            ),
            description_placeholders={CONF_NAME: current_name}, # Pass current_name for use in translation string
            # self._set_confirm_only() is usually for forms with no user input fields, just a confirm button.
            # Since we added a name field, it's a regular form step now.
        )


from homeassistant.config_entries import ConfigEntry, OptionsFlow # Ensure ConfigEntry and OptionsFlow are imported
# ... (other imports might be here) ...

class BookooOptionsFlowHandler(OptionsFlow):
    """Handle an options flow for Bookoo."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        # self.config_entry is automatically available from the base class
        # Or, if you need to manipulate options, you might do:
        # self.options = dict(config_entry.options)
        # For this fix, we are just removing the deprecated assignment.

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Basic validation example (can be expanded)
            min_duration = user_input.get(OPTION_MIN_SHOT_DURATION, 0)
            if not isinstance(min_duration, (int, float)) or min_duration < 0:
                errors[OPTION_MIN_SHOT_DURATION] = "invalid_duration_positive_number_expected"
            
            if not errors:
                return self.async_create_entry(title="", data=user_input)

        # Get current options to pre-fill the form
        current_min_duration = self.config_entry.options.get(OPTION_MIN_SHOT_DURATION, 10) # Default to 10s
        current_bean_weight_entity = self.config_entry.options.get(OPTION_LINKED_BEAN_WEIGHT_ENTITY)
        current_coffee_name_entity = self.config_entry.options.get(OPTION_LINKED_COFFEE_NAME_ENTITY)

        options_schema = vol.Schema(
            {
                vol.Optional(
                    OPTION_MIN_SHOT_DURATION,
                    default=current_min_duration,
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0, mode=NumberSelectorMode.BOX, unit_of_measurement="s", step=1
                    )
                ),
                vol.Optional(
                    OPTION_LINKED_BEAN_WEIGHT_ENTITY,
                    default=current_bean_weight_entity if current_bean_weight_entity else vol.UNDEFINED,
                ): EntitySelector(
                    EntitySelectorConfig(domain="input_number", multiple=False)
                ),
                vol.Optional(
                    OPTION_LINKED_COFFEE_NAME_ENTITY,
                    default=current_coffee_name_entity if current_coffee_name_entity else vol.UNDEFINED,
                ): EntitySelector(
                    EntitySelectorConfig(domain="input_text", multiple=False)
                ),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,
            description_placeholders={ # Optional: provide descriptions for fields
                OPTION_MIN_SHOT_DURATION: "Minimum duration for a shot to be considered valid.",
                OPTION_LINKED_BEAN_WEIGHT_ENTITY: "Select an input_number entity for bean weight.",
                OPTION_LINKED_COFFEE_NAME_ENTITY: "Select an input_text entity for coffee name/type."
            }
        )
