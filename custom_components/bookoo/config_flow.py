"""Config flow for Bookoo integration."""

import logging
from typing import Any

from aiobookoo.exceptions import BookooDeviceNotFound, BookooError, BookooUnknownDevice
from aiobookoo.helpers import is_bookoo_scale
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

from .const import CONF_IS_VALID_SCALE, DOMAIN

_LOGGER = logging.getLogger(__name__)


# Define constants for option keys to avoid typos
OPTION_MIN_SHOT_DURATION = "minimum_shot_duration_seconds"
OPTION_LINKED_BEAN_WEIGHT = "linked_bean_weight_entity"
OPTION_LINKED_COFFEE_NAME = "linked_coffee_name_entity"
# Add more as needed for other linked inputs


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
            return self.async_create_entry(
                title=self._discovered[CONF_NAME],
                data={
                    CONF_ADDRESS: self._discovered[CONF_ADDRESS],
                    CONF_IS_VALID_SCALE: self._discovered[CONF_IS_VALID_SCALE],
                },
            )

        self.context["title_placeholders"] = placeholders = {
            CONF_NAME: self._discovered[CONF_NAME]
        }

        self._set_confirm_only()
        return self.async_show_form(
            step_id="bluetooth_confirm",
            description_placeholders=placeholders,
        )


class BookooOptionsFlowHandler(OptionsFlow):
    """Handle an options flow for Bookoo."""

    def __init__(
        self, config_entry: ConfigFlowResult
    ) -> None:  # ConfigEntry is more appropriate
        """Initialize options flow."""
        self.config_entry = config_entry
        # Or, more commonly, just store the options directly:
        # self.options = dict(config_entry.options)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validation can be added here if needed
            # For example, check if min_shot_duration is within a reasonable range
            # if user_input.get(OPTION_MIN_SHOT_DURATION, 5) < 0:
            #     errors["base"] = "min_shot_duration_negative"
            #     # Or specifically: errors[OPTION_MIN_SHOT_DURATION] = "value_too_low"

            if not errors:
                # self.options.update(user_input) # If storing options in self.options
                # return self.async_create_entry(title="", data=self.options)
                return self.async_create_entry(title="", data=user_input)

        # Define the schema for the options form
        schema = vol.Schema(
            {
                vol.Optional(
                    OPTION_MIN_SHOT_DURATION,
                    default=self.config_entry.options.get(OPTION_MIN_SHOT_DURATION, 5),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0, max=60, step=1, mode=NumberSelectorMode.SLIDER
                    )
                ),
                vol.Optional(
                    OPTION_LINKED_BEAN_WEIGHT,
                    default=self.config_entry.options.get(
                        OPTION_LINKED_BEAN_WEIGHT, ""
                    ),
                ): EntitySelector(
                    EntitySelectorConfig(domain="input_number", multiple=False)
                ),
                vol.Optional(
                    OPTION_LINKED_COFFEE_NAME,
                    default=self.config_entry.options.get(
                        OPTION_LINKED_COFFEE_NAME, ""
                    ),
                ): EntitySelector(
                    EntitySelectorConfig(domain="input_text", multiple=False)
                ),
                # Add more EntitySelectors for other parameters as needed
                # e.g., grind setting (input_number or input_text), roast date (input_datetime), notes (input_text)
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
            # last_step=True # if this is the only options step
        )
