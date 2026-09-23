import objc
import warnings

# PyObjC emits a harmless warning when CALayer receives a CGColor pointer.
# The color is intentionally passed to the layer; suppress only that warning.
warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)

import json
import re
import threading
import urllib.request
import urllib.error
import time
import os

import sounddevice as sd

from PyObjCTools import AppHelper
from assistant_functionality import AIService, AudioService

from Cocoa import (
    NSTextStorage,
    NSTextContainer,
    NSLayoutManager,
    NSApplication,
    NSWindow,
    NSTextField,
    NSTextView,
    NSScrollView,
    NSButton,
    NSImage,
    NSView,
    NSColor,
    NSFont,
    NSMakeRect,
    NSMakeSize,
    NSImageScaleProportionallyUpOrDown,
    NSBackingStoreBuffered,
    NSFloatingWindowLevel,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskResizable,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowSharingNone,
    NSBezelStyleRounded,
    NSEventModifierFlagCommand,
    NSEventModifierFlagOption,
    NSEventMaskKeyDown,
    NSEventMaskFlagsChanged,
    NSEventTypeKeyDown,
    NSEventTypeFlagsChanged,
    NSEvent
)

from Foundation import NSObject, NSMutableAttributedString


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MODEL = "llama3.2:3b"
VISION_MODEL = "llava:7b"

NUM_CTX = 1024
NUM_PREDICT = 120

TEMPERATURE = 0.1

UI_UPDATE_INTERVAL = 0.15


# ============================================================
# WHISPER CONFIGURATION
# ============================================================

# Good balance between speed and accuracy.
#
# tiny.en  -> fastest
# base.en  -> better accuracy
# small.en -> better accuracy but slower
#
# Start with base.en.
WHISPER_MODEL = "base.en"

SAMPLE_RATE = 16000

CHANNELS = 1

# Maximum recording time.
MAX_RECORD_SECONDS = 30


# ============================================================
# CODE POPUP
# ============================================================

class CodeTextView(NSTextView):

    def keyDown_(self, event):

        controller = getattr(
            self,
            "controller",
            None,
        )

        if controller is None:
            objc.super(
                CodeTextView,
                self,
            ).keyDown_(event)

            return

        key_code = event.keyCode()

        flags = event.modifierFlags()

        command_pressed = bool(
            flags & NSEventModifierFlagCommand
        )

        # UP
        if key_code == 126:

            if command_pressed:
                controller.move_popup(
                    0,
                    60,
                )
            else:
                controller.scroll_popup(
                    -80
                )

            return

        # DOWN
        if key_code == 125:

            if command_pressed:
                controller.move_popup(
                    0,
                    -60,
                )
            else:
                controller.scroll_popup(
                    80
                )

            return

        # LEFT
        if key_code == 123:

            controller.move_popup(
                -60,
                0,
            )

            return

        # RIGHT
        if key_code == 124:

            controller.move_popup(
                60,
                0,
            )

            return

        # ESC
        if key_code == 53:

            controller.close_popup()

            return

        objc.super(
            CodeTextView,
            self,
        ).keyDown_(event)


class PushToTalkButton(NSButton):
    """
    Display-only microphone status button.

    Recording is controlled exclusively by the right Option + right Command
    keyboard combination. No mouse press/release is used.
    """

    def cancel_push_to_talk(self):
        controller = getattr(self, "controller", None)
        if controller is not None and controller.recording:
            controller.stop_recording()



# ============================================================
# APPLICATION
# ============================================================

class AppDelegate(NSObject):

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def applicationDidFinishLaunching_(
        self,
        notification,
    ):

        self.popup_window = None
        self.popup_code_view = None
        self.popup_scroll = None

        # Main-window keyboard movement state.
        # Hold Option + Arrow keys to move the main window.
        self._window_move_monitor = None
        self._window_move_step = 20

        self.response_expanded = True
        self.code_expanded = True

        # Automatic response-window growth.
        # Compact -> response -> auto-grow sizing.
        self.compact_window_width = 760
        self.compact_window_height = 320
        self.response_window_min_height = 320
        self.response_window_max_height = 1200
        self.response_min_window_height = 320
        self.response_max_window_height = 900
        self.has_answer = False

        # Modern minimal UI sizing.
        self.content_side_padding = 24
        self.response_card_padding = 16
        self.response_card_min_height = 110
        self.response_card_max_height = 1060

        self.recording = False
        self.ptt_event_monitor = None
        self.ptt_local_event_monitor = None
        self.audio_data = None

        self.whisper_model = None
        self.ai_service = AIService(OLLAMA_URL, MODEL, VISION_MODEL)
        self.audio_service = AudioService(WHISPER_MODEL)


        self.create_main_window()
        self.start_window_arrow_move()

        # Load Whisper in background.
        threading.Thread(
            target=self.load_whisper,
            daemon=True,
        ).start()

    # ========================================================
    # LOAD WHISPER
    # ========================================================

    @objc.python_method
    def load_whisper(self):
        try:
            AppHelper.callAfter(self.set_status, "● LOADING VOICE AI...", NSColor.systemOrangeColor())
            self.whisper_model = self.audio_service.load()
            AppHelper.callAfter(self.set_status, "● LOCAL AI", NSColor.systemGreenColor())
        except Exception as error:
            AppHelper.callAfter(self.show_error, f"Unable to load Whisper.\n\n{error}")

    # ========================================================
    # MAIN WINDOW
    # ========================================================

    @objc.python_method
    def create_main_window(self):

        # Start compact: prompt-only UI.
        frame = NSMakeRect(
            0,
            0,
            760,
            320,
        )

        # Native title bar keeps close/minimize/resize controls visible.
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskResizable
        )

        self.window = (
            NSWindow.alloc()
            .initWithContentRect_styleMask_backing_defer_(
                frame,
                style,
                NSBackingStoreBuffered,
                False,
            )
        )

        self.window.setTitle_(" AI Assistant")
        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setDelegate_(self)

        self.window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Keep overlay excluded from supported capture.
        self.window.setSharingType_(NSWindowSharingNone)

        # Modern dark application surface.
        self.window.setOpaque_(True)
        self.window.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.055, 0.065, 0.085, 1.0
            )
        )

        self.window.center()

        content = self.window.contentView()

        # ====================================================
        # CUSTOM NAV BAR
        # ====================================================

        self.nav_bar = (
            NSView.alloc()
            .initWithFrame_(
                NSMakeRect(0, 700, 760, 60)
            )
        )

        self.nav_bar.setWantsLayer_(True)
        self.nav_bar.layer().setBackgroundColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
        0.10,
        0.20,
        0.15,
        1.0
    ).CGColor()
        )
        content.addSubview_(self.nav_bar)

        # Small local-AI indicator instead of an application title.
        self.nav_status = (
            NSTextField.alloc()
            .initWithFrame_(
                NSMakeRect(90, 18, 130, 25)
            )
        )
        self.nav_status.setStringValue_("AI READY")
        self.nav_status.setFont_(
            NSFont.boldSystemFontOfSize_(14)
        )
        self.nav_status.setTextColor_(
            NSColor.blackColor()
        )
        self.nav_status.setDrawsBackground_(False)
        self.nav_status.setBezeled_(False)
        self.nav_status.setEditable_(False)
        self.nav_bar.addSubview_(self.nav_status)

        # Compact Quit button beside the live/status indicator.
        self.quit_button = NSButton.alloc().initWithFrame_(
            NSMakeRect(0, 0, 64, 32)
        )
        self.quit_button.setTitle_("Quit")
        self.quit_button.setFont_(NSFont.boldSystemFontOfSize_(12))
        self.quit_button.setToolTip_("Quit Private AI Assistant")
        self.quit_button.setBezelStyle_(NSBezelStyleRounded)
        self.quit_button.setBordered_(True)
        self.quit_button.setContentTintColor_(NSColor.whiteColor())
        self.quit_button.setTarget_(self)
        self.quit_button.setAction_("closeApp:")
        self.nav_bar.addSubview_(self.quit_button)

        def nav_button(x, title, tooltip, action):
            button = NSButton.alloc().initWithFrame_(
                NSMakeRect(x, 8, 82, 44)
            )
            button.setTitle_(title)
            button.setFont_(NSFont.boldSystemFontOfSize_(12))
            button.setToolTip_(tooltip)
            button.setAlignment_(1)
            button.setBezelStyle_(NSBezelStyleRounded)
            button.setBordered_(True)
            button.setContentTintColor_(NSColor.blackColor())
            button.setTarget_(self)
            button.setAction_(action)
            self.nav_bar.addSubview_(button)
            return button

        # Named buttons are intentionally used instead of SF Symbols so the
        # controls are immediately understandable and render consistently.
        # Microphone status button.
        # Recording is controlled by Right Option + Right Command.
        self.nav_mic_button = PushToTalkButton.alloc().initWithFrame_(
            NSMakeRect(250, 8, 110, 42)
        )
        self.nav_mic_button.setTitle_("Microphone")
        self.nav_mic_button.setFont_(
            NSFont.boldSystemFontOfSize_(12)
        )
        self.nav_mic_button.setToolTip_(
            "Hold Right Option + Right Command to record"
        )
        self.nav_mic_button.setAlignment_(1)
        self.nav_mic_button.setBezelStyle_(NSBezelStyleRounded)
        self.nav_mic_button.setBordered_(True)
        self.nav_mic_button.setContentTintColor_(NSColor.blackColor())
        self.nav_mic_button.controller = self
        self.nav_bar.addSubview_(self.nav_mic_button)


        self.nav_screenshot_button = nav_button(
            338,
            "Screenshot",
            "Capture screen and answer",
            "processScreenshot:",
        )

        self.nav_code_button = nav_button(
            426,
            "Code",
            "Show / hide code",
            "toggleCode:",
        )

        self.nav_definition_button = nav_button(
            514,
            "Definition",
            "Show definition",
            "showDefinition:",
        )

        self.nav_view_button = nav_button(
            602,
            "View",
            "View response only",
            "toggleViewOnly:",
        )

        self.status = (
            NSTextField.alloc()
            .initWithFrame_(
                NSMakeRect(610, 18, 110, 25)
            )
        )
        self.status.setStringValue_(
            "● READY"
        )
        self.status.setFont_(
            NSFont.boldSystemFontOfSize_(11)
        )
        self.status.setTextColor_(
            NSColor.systemGreenColor()
        )
        self.status.setAlignment_(2)
        self.status.setDrawsBackground_(False)
        self.status.setBezeled_(False)
        self.status.setEditable_(False)
        self.nav_bar.addSubview_(self.status)

        self.screenshot_button = self.nav_screenshot_button

        self.view_only = False
        self.definition_only = False
        self._definition_backup = None

        # ====================================================
        # PROMPT
        # ====================================================

        self.ask_label = (
            NSTextField.alloc()
            .initWithFrame_(
                NSMakeRect(25, 650, 100, 22)
            )
        )
        self.ask_label.setStringValue_("Ask AI")
        self.ask_label.setFont_(
            NSFont.boldSystemFontOfSize_(13)
        )
        self.ask_label.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.90, 0.92, 0.96, 1.0
        ))
        self.ask_label.setDrawsBackground_(False)
        self.ask_label.setBezeled_(False)
        self.ask_label.setEditable_(False)
        content.addSubview_(self.ask_label)

        self.prompt = (
            NSTextField.alloc()
            .initWithFrame_(
                NSMakeRect(25, 610, 695, 38)
            )
        )
        self.prompt.setFont_(
            NSFont.systemFontOfSize_(15)
        )
        self.prompt.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.90, 0.92, 0.96, 1.0
            )
        )
        self.prompt.setDrawsBackground_(True)
        self.prompt.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.09, 0.10, 0.13, 1.0
            )
        )
        self.prompt.setPlaceholderString_(
            "Type or speak your question..."
        )
        self.prompt.setTarget_(self)
        self.prompt.setAction_("ask:")
        content.addSubview_(self.prompt)

        # Existing logic uses these references; point them to the nav icons.
        self.mic_button = self.nav_mic_button
        self.ask_button = self.nav_screenshot_button

        # ====================================================
        # RESPONSE HEADER
        # ====================================================

        self.response_header = (
            NSButton.alloc()
            .initWithFrame_(
                NSMakeRect(25, 575, 695, 32)
            )
        )
        self.response_header.setTitle_(
            "AI RESPONSE"
        )
        self.response_header.setFont_(
            NSFont.systemFontOfSize_weight_(12, 0.55)
        )
        self.response_header.setAlignment_(0)
        self.response_header.setBordered_(False)
        self.response_header.setContentTintColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.60, 0.65, 0.72, 1.0
            )
        )
        self.response_header.setTarget_(self)
        self.response_header.setAction_("toggleResponse:")
        content.addSubview_(self.response_header)

        # ====================================================
        # RESPONSE
        # ====================================================

        self.response_scroll = (
            NSScrollView.alloc()
            .initWithFrame_(
                NSMakeRect(24, 345, 695, 225)
            )
        )
        self.response_scroll.setHasVerticalScroller_(True)
        self.response_scroll.setHasHorizontalScroller_(False)
        self.response_scroll.setBorderType_(0)
        # Never use the default white NSScrollView background.
        self.response_scroll.setDrawsBackground_(True)
        self.response_scroll.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.075, 0.085, 0.105, 1.0
            )
        )
        self.response_scroll.setAutohidesScrollers_(True)
        self.response_scroll.setScrollerStyle_(1)

        self.response_view = (
            NSTextView.alloc()
            .initWithFrame_(
                NSMakeRect(0, 0, 675, 220)
            )
        )
        self.response_view.setEditable_(False)
        self.response_view.setSelectable_(True)
        self.response_view.setRichText_(True)
        self.response_view.setDrawsBackground_(False)
        # Clean white response text on the dark response surface.
        self.response_view.setTextColor_(
            NSColor.colorWithCalibratedWhite_alpha_(
                0.97, 1.0
            )
        )
        self.response_view.setFont_(
            NSFont.systemFontOfSize_(16)
        )
        self.response_view.setTextContainerInset_(
            NSMakeSize(16, 14)
        )
        self.response_view.textContainer().setLineFragmentPadding_(0)
        self.response_view.setHorizontallyResizable_(False)
        self.response_view.setVerticallyResizable_(True)
        self.response_view.setAutoresizingMask_(1 | 16)
        self.response_scroll.setDocumentView_(
            self.response_view
        )
        content.addSubview_(self.response_scroll)

        # ====================================================
        # CODE HEADER
        # ====================================================

        self.code_header = (
            NSButton.alloc()
            .initWithFrame_(
                NSMakeRect(25, 305, 695, 32)
            )
        )
        self.code_header.setTitle_("CODE")
        self.code_header.setFont_(
            NSFont.boldSystemFontOfSize_(14)
        )
        self.code_header.setAlignment_(0)
        self.code_header.setBordered_(False)
        self.code_header.setContentTintColor_(
            NSColor.blackColor()
        )
        self.code_header.setTarget_(self)
        self.code_header.setAction_("toggleCode:")
        content.addSubview_(self.code_header)

        # ====================================================
        # CODE
        # ====================================================

        self.code_scroll = (
            NSScrollView.alloc()
            .initWithFrame_(
                NSMakeRect(25, 105, 695, 190)
            )
        )
        self.code_scroll.setHasVerticalScroller_(True)
        self.code_scroll.setHasHorizontalScroller_(True)

        self.code_view = (
            NSTextView.alloc()
            .initWithFrame_(
                NSMakeRect(0, 0, 680, 180)
            )
        )
        self.code_view.setEditable_(False)
        self.code_view.setSelectable_(True)
        self.code_view.setFont_(
            NSFont.monospacedSystemFontOfSize_weight_(13, 0.0)
        )
        self.code_view.setTextColor_(
            NSColor.blackColor()
        )
        self.code_view.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.055, 0.065, 0.085, 1.0
            )
        )
        self.code_view.setRichText_(False)
        self.code_scroll.setDocumentView_(self.code_view)
        content.addSubview_(self.code_scroll)

        # Secondary controls are hidden until the first answer is ready.
        # Code is always included in the answer; no separate Code navigation.
        self.nav_code_button.setHidden_(True)
        self.nav_definition_button.setHidden_(True)
        self.nav_view_button.setHidden_(True)
        self.nav_definition_button.setHidden_(True)
        self.nav_view_button.setHidden_(True)
        self.code_header.setHidden_(True)
        self.code_scroll.setHidden_(True)

        # Old bottom controls are no longer used.
        self.more_button = None
        self.code_button = None

        self.response_view.setString_(
            "Definition:\n\n"
            "Ask a question by typing or holding the microphone button, "
            "or capture the screen for visual understanding.\n\n"
            "Key Points:\n\n"
            "• Microphone — speak a question.\n"
            "• Camera — analyze one screenshot.\n"
            "• Code — show or hide generated code.\n"
            "• Definition — show the definition section.\n"
            "• Eye — view-only mode."
        )

        self.code_view.setString_("No code required.")

        self.layout_main_ui()

        # Option + Command keyboard push-to-talk.
        self.install_push_to_talk_monitor()

        # Explicitly bring the main window to the foreground on startup.
        self.window.orderFrontRegardless()
        self.window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    # ========================================================
    # WINDOW / RESPONSIVE UI
    # ========================================================

    def windowWillResize_toSize_(self, window, size):
        # Keep the UI usable at all times.
        return NSMakeSize(
            max(680.0, size.width),
            max(320.0, size.height),
        )

    def windowDidResize_(self, notification):
        self.layout_main_ui()

    @objc.python_method
    def _set_nav_icon(self, button, symbol_name, fallback):
        # Kept under the old method name for compatibility with the existing
        # functionality. The UI now uses readable text labels instead of icons.
        button.setImage_(None)
        button.setTitle_(fallback)
        button.setFont_(NSFont.boldSystemFontOfSize_(12))

    @objc.python_method
    def layout_main_ui(self):
        """Deterministic responsive layout for the modern minimal UI."""

        if not hasattr(self, "window"):
            return

        bounds = self.window.contentView().bounds()
        w = max(680.0, bounds.size.width)
        h = max(320.0, bounds.size.height)

        pad = 24.0
        nav_h = 54.0

        # --------------------------------------------------------
        # NAV BAR
        # --------------------------------------------------------
        self.nav_bar.setFrame_(
            NSMakeRect(0, h - nav_h, w, nav_h)
        )

        self.nav_status.setFrame_(
            NSMakeRect(18, 15, 140, 24)
        )

        self.status.setFrame_(
            NSMakeRect(w - 190, 15, 105, 24)
        )
        if hasattr(self, "quit_button"):
            # Place Quit directly beside the live/status indicator.
            status_frame = self.status.frame()
            self.quit_button.setFrame_(
                NSMakeRect(
                    status_frame.origin.x + status_frame.size.width + 8,
                    11,
                    64,
                    32,
                )
            )

        buttons = [
            self.nav_mic_button,
            self.nav_screenshot_button,
            self.nav_code_button,
            self.nav_definition_button,
            self.nav_view_button,
        ]

        visible = [
            button for button in buttons
            if not button.isHidden()
        ]

        button_widths = {
            "Microphone": 94,
            "Screenshot": 94,
            "Code": 64,
            "Definition": 92,
            "View": 64,
        }

        gap = 5.0
        total_width = sum(
            button_widths.get(str(button.title()), 80)
            for button in visible
        )
        total_width += max(0, len(visible) - 1) * gap

        left_limit = 155.0
        right_limit = w - 125.0

        if total_width <= max(0.0, right_limit - left_limit):
            x = max(left_limit, (w - total_width) / 2.0)

            for button in visible:
                width = button_widths.get(
                    str(button.title()),
                    80,
                )
                button.setFrame_(
                    NSMakeRect(x, 6, width, 42)
                )
                x += width + gap

        # --------------------------------------------------------
        # PROMPT
        # --------------------------------------------------------
        prompt_y = h - nav_h - 112.0

        self.ask_label.setFrame_(
            NSMakeRect(
                pad,
                prompt_y + 42,
                100,
                20,
            )
        )

        self.prompt.setFrame_(
            NSMakeRect(
                pad,
                prompt_y,
                w - (2.0 * pad),
                40,
            )
        )

        if not self.has_answer:
            self.response_header.setHidden_(True)
            self.response_scroll.setHidden_(True)
            self.code_header.setHidden_(True)
            self.code_scroll.setHidden_(True)
            return

        # --------------------------------------------------------
        # RESPONSE CARD
        # --------------------------------------------------------
        self.response_header.setHidden_(False)

        header_y = prompt_y - 42.0

        self.response_header.setFrame_(
            NSMakeRect(
                pad,
                header_y,
                w - (2.0 * pad),
                30,
            )
        )

        # The response card always occupies the space between the
        # header and the bottom padding. Window height is changed by
        # grow_response_window(), not by this method.
        response_y = 24.0
        response_h = max(
            self.response_card_min_height,
            header_y - response_y - 10.0,
        )

        self.response_scroll.setFrame_(
            NSMakeRect(
                pad,
                response_y,
                w - (2.0 * pad),
                response_h,
            )
        )
        self.response_scroll.setHidden_(False)

        # The document view width must exactly match the scroll view's
        # content width. Height is controlled by grow_response_window().
        document_width = max(
            200.0,
            w - (2.0 * pad) - 4.0,
        )

        current_doc_height = max(
            response_h,
            self.response_view.frame().size.height,
        )

        self.response_view.setFrameSize_(
            NSMakeSize(
                document_width,
                current_doc_height,
            )
        )

        self.response_view.textContainer().setContainerSize_(
            NSMakeSize(document_width - 4.0, 1000000.0)
        )

        self.code_header.setHidden_(True)
        self.code_scroll.setHidden_(True)

    # ========================================================
    # MOVE MAIN WINDOW WITH OPTION + ARROW KEYS
    # ========================================================

    @objc.python_method
    def start_window_arrow_move(self):
        """Hold Option and press arrow keys to move the main window."""
        if self._window_move_monitor is not None:
            return

        mask = NSEventMaskKeyDown

        def handle_key(event):
            try:
                flags = event.modifierFlags()
                option_pressed = bool(
                    flags & NSEventModifierFlagOption
                )

                if not option_pressed:
                    return event

                key_code = event.keyCode()

                # macOS arrow key codes:
                # Left=123, Right=124, Down=125, Up=126
                dx = 0
                dy = 0

                if key_code == 123:       # Left
                    dx = -self._window_move_step
                elif key_code == 124:     # Right
                    dx = self._window_move_step
                elif key_code == 125:     # Down
                    dy = -self._window_move_step
                elif key_code == 126:     # Up
                    dy = self._window_move_step
                else:
                    return event

                self.move_main_window(dx, dy)

                # Returning None consumes Option + Arrow so it does
                # not also perform another action in the application.
                return None

            except Exception:
                return event

        self._window_move_monitor = (
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                mask,
                handle_key,
            )
        )

    @objc.python_method
    def move_main_window(self, dx, dy):
        """Move the main application window by dx/dy pixels."""
        if not hasattr(self, "window") or self.window is None:
            return

        frame = self.window.frame()

        frame.origin.x += dx
        frame.origin.y += dy

        self.window.setFrameOrigin_(frame.origin)

    @objc.python_method
    def stop_window_arrow_move(self):
        if self._window_move_monitor is not None:
            NSEvent.removeMonitor_(self._window_move_monitor)
            self._window_move_monitor = None

    def windowWillClose_(self, notification):

        self.remove_push_to_talk_monitor()
        self.stop_window_arrow_move()

        try:
            if False:
                        None.stop()
        except Exception:
            pass
        try:
            if self.recording:
                self.recording = False
        except Exception:
            pass

    def closeApp_(self, sender):
        self.nav_mic_button.cancel_push_to_talk()

        self.remove_push_to_talk_monitor()
        self.stop_window_arrow_move()

        try:
            if False:
                        None.stop()
        except Exception:
            pass
        try:
            if self.recording:
                self.recording = False
        except Exception:
            pass
        NSApplication.sharedApplication().terminate_(self)

    # ========================================================
    # STATUS
    # ========================================================

    def set_status(
        self,
        text,
        color,
    ):

        self.status.setStringValue_(
            text
        )

        self.status.setTextColor_(
            color
        )

    @objc.python_method
    def show_answer_controls(self):
        # Keep the response simple: no separate Code/Definition/View navigation.
        self.nav_code_button.setHidden_(True)
        self.nav_definition_button.setHidden_(True)
        self.nav_view_button.setHidden_(True)
        self.layout_main_ui()

    # ========================================================
    # SCREENSHOT -> LLAVA -> LLAMA
    # ========================================================

    def processScreenshot_(self, sender):
        if self.screenshot_processing:
            return
        self.screenshot_processing = True
        self.screenshot_button.setEnabled_(False)
        self.mic_button.setEnabled_(True)
        self.set_status("● CAPTURING SCREEN...", NSColor.systemOrangeColor())
        self.response_view.setString_("Capturing screen...")
        self.window.orderOut_(None)
        threading.Thread(target=self._capture_and_process_screen, daemon=True).start()

    @objc.python_method
    def _capture_and_process_screen(self):
        try:
            AppHelper.callAfter(self._screen_status, "● ANALYZING SCREEN...")
            answer, context = self.ai_service.capture_and_answer()
            AppHelper.callAfter(self._show_screen_answer, answer, context)
        except Exception as error:
            AppHelper.callAfter(self.show_error, f"Screen AI error:\n\n{error}")
        finally:
            AppHelper.callAfter(self._finish_screen_processing)

    @objc.python_method
    def _screen_status(self, text):
        self.set_status(text, NSColor.systemOrangeColor())

    @objc.python_method
    def _show_screen_answer(self, answer, screen_context):
        self.response_view.setString_(answer.strip())
        self._set_response_text_white()
        self.code_view.setString_(screen_context.strip())
        self.has_answer = True
        self.grow_response_window()
        self.show_answer_controls()
        self.set_status("● SCREEN AI", NSColor.systemGreenColor())

    @objc.python_method
    def _finish_screen_processing(self):
        self.window.makeKeyAndOrderFront_(None)
        self.screenshot_button.setEnabled_(True)
        self.mic_button.setEnabled_(True)
        self.screenshot_processing = False


    # ========================================================
    # MICROPHONE
    # ========================================================

    def toggleRecording_(self, sender):

        if self.recording:

            self.stop_recording()

        else:

            self.start_recording()

    # ========================================================
    # START RECORDING
    # ========================================================

    @objc.python_method
    def install_push_to_talk_monitor(self):
        """
        Right-side keyboard push-to-talk.

        Right Option keyCode 61
        Right Command keyCode 54

        Press both -> start recording.
        Release either -> stop recording and process the response.
        """

        if self.ptt_event_monitor is not None:
            return

        def handler(event):
            flags = event.modifierFlags()
            key_code = event.keyCode()

            # Right Option = 61, Right Command = 54.
            # Ignore left-side modifier changes.
            if key_code not in (54, 61):
                return event

            right_option = bool(flags & NSEventModifierFlagOption)
            right_command = bool(flags & NSEventModifierFlagCommand)

            if right_option and right_command:
                if not self.recording:
                    self.start_recording()
            else:
                if self.recording:
                    self.stop_recording()

            return event

        self.ptt_event_monitor = (
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                NSEventMaskFlagsChanged,
                handler,
            )
        )

        # Also monitor events generated while this application is active.
        self.ptt_local_event_monitor = (
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                NSEventMaskFlagsChanged,
                handler,
            )
        )

    def remove_push_to_talk_monitor(self):
        if self.ptt_event_monitor is not None:
            NSEvent.removeMonitor_(self.ptt_event_monitor)
            self.ptt_event_monitor = None

        if getattr(self, "ptt_local_event_monitor", None) is not None:
            NSEvent.removeMonitor_(self.ptt_local_event_monitor)
            self.ptt_local_event_monitor = None


    def start_recording(self):
        """
        Start push-to-talk recording immediately.

        Whisper is intentionally NOT checked here. The user should be able
        to hold the microphone button for exactly as long as required.
        Whisper is loaded during transcription after release.
        """

        if self.recording:
            return

        self.recording = True
        self.audio_data = None
        self.audio_chunks = []

        self._set_nav_icon(
            self.mic_button,
            "mic.fill",
            "● Recording...",
        )

        self.status.setStringValue_("● LISTENING...")
        self.status.setTextColor_(NSColor.systemRedColor())

        self.response_view.setString_(
            "Listening...\n\nRelease the microphone button when you finish speaking."
        )

        threading.Thread(
            target=self.record_audio,
            daemon=True,
        ).start()


    # ========================================================
    # RECORD AUDIO
    # ========================================================

    @objc.python_method
    def record_audio(self):
        """Capture audio continuously until the push-to-talk control is released."""

        stream = None

        try:
            self.audio_chunks = []

            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio input status: {status}")

                if self.recording:
                    self.audio_chunks.append(indata.copy())

            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                callback=audio_callback,
                blocksize=0,
            )

            self.audio_stream = stream
            stream.start()

            started_at = time.monotonic()

            while self.recording:
                # Safety limit.
                if time.monotonic() - started_at >= MAX_RECORD_SECONDS:
                    AppHelper.callAfter(
                        self._max_recording_reached
                    )
                    break

                time.sleep(0.02)

            if stream is not None:
                stream.stop()
                stream.close()
                stream = None

            self.audio_stream = None

            if self.audio_chunks:
                try:
                    import numpy as np
                    self.audio_data = np.concatenate(
                        self.audio_chunks,
                        axis=0,
                    )
                except Exception:
                    self.audio_data = self.audio_chunks[0]
            else:
                self.audio_data = None

            if self.audio_data is not None:
                AppHelper.callAfter(
                    self.transcribe_audio,
                )
            else:
                AppHelper.callAfter(
                    self.show_error,
                    "No audio was captured. Check your microphone permission.",
                )

        except Exception as error:
            self.recording = False
            self.audio_stream = None

            try:
                if stream is not None:
                    stream.stop()
                    stream.close()
            except Exception:
                pass

            AppHelper.callAfter(
                self.show_error,
                (
                    "Microphone error:\n\n"
                    f"{error}"
                ),
            )

    def _max_recording_reached(self):
        if self.recording:
            self.stop_recording()


    # ========================================================
    # STOP RECORDING
    # ========================================================

    @objc.python_method
    def stop_recording(self):

        if not self.recording:
            return

        self.recording = False

        # InputStream is stopped by record_audio after seeing recording=False.
        # Do not call sd.stop() here because it can interfere with other
        # CoreAudio streams on macOS.
        self._set_nav_icon(self.mic_button, "mic.fill", "Microphone")

        self.status.setStringValue_(
            "● TRANSCRIBING..."
        )

        self.status.setTextColor_(
            NSColor.systemOrangeColor()
        )

    # ========================================================
    # TRANSCRIBE
    # ========================================================

    @objc.python_method
    def transcribe_audio(self):
        try:
            if self.audio_data is None:
                self.show_error(
                    "No audio was captured. Please hold the Microphone button while speaking."
                )
                return

            self.status.setStringValue_("● LOADING VOICE AI...")

            # Lazy-load Whisper after recording. This avoids loading the
            # native Whisper stack during application startup.
            if self.whisper_model is None:
                self.whisper_model = self.audio_service.load()

            self.status.setStringValue_("● TRANSCRIBING...")

            text = self.audio_service.transcribe(
                self.audio_data,
                SAMPLE_RATE,
            )

            text = (text or "").strip()

            if not text:
                self.show_error(
                    "No speech detected. Please hold the Microphone button and speak clearly."
                )
                return

            self.process_transcription(text)

        except Exception as error:
            self.show_error(
                f"Unable to transcribe audio.\n\n{error}"
            )


    def process_transcription(
        self,
        text,
    ):

        if not text:

            self.response_view.setString_(
                "I couldn't understand the audio.\n\n"
                "Please try speaking again."
            )

            self.set_status(
                "● LOCAL AI",
                NSColor.systemGreenColor(),
            )

            return

        # Put recognized speech in input box.
        self.prompt.setStringValue_(
            text
        )

        # Automatically send to Llama.
        self.ask_question(
            text
        )

    # ========================================================
    # ASK BUTTON / ENTER
    # ========================================================

    def ask_(self, sender):

        question = (
            self.prompt.stringValue()
            .strip()
        )

        if not question:
            return

        self.ask_question(
            question
        )

    # ========================================================
    # ASK QUESTION
    # ========================================================

    @objc.python_method
    def ask_question(
        self,
        question,
    ):

        self.ask_button.setEnabled_(
            False
        )

        self.mic_button.setEnabled_(
            False
        )

        self.status.setStringValue_(
            "● THINKING..."
        )

        self.status.setTextColor_(
            NSColor.systemOrangeColor()
        )

        self.has_answer = False

        self.response_view.setString_(
            "Generating..."
        )

        self.code_view.setString_(
            "Waiting for response..."
        )

        # Return to the compact prompt-only layout while waiting.
        self.response_header.setHidden_(True)
        self.response_scroll.setHidden_(True)
        self.code_header.setHidden_(True)
        self.code_scroll.setHidden_(True)

        frame = self.window.frame()
        target_height = self.compact_window_height

        if frame.size.height > target_height:
            frame.origin.y += (
                frame.size.height - target_height
            )
            frame.size.height = target_height
            self.window.setFrame_display_animate_(
                frame,
                True,
                False,
            )

        self.layout_main_ui()

        thread = threading.Thread(
            target=self.ask_ollama,
            args=(question,),
            daemon=True,
        )

        thread.start()

    # ========================================================
    # OLLAMA
    # ========================================================

    @objc.python_method
    def ask_ollama(
        self,
        question,
    ):

        prompt = f"""Answer this question concisely.

Question:
{question}

Format:

Definition:
1-2 clear sentences.

Key Points:
2-4 important bullets.

Explanation:
Only if necessary.

Rules:
- Be accurate.
- Be direct.
- No repetition.
- No unnecessary introduction.
- Maximum 80 words.
- If code is required, put ALL code in ONE markdown code block.
- Never put code outside the code block.
"""

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": True,

            # Keep model loaded.
            "keep_alive": -1,

            "options": {
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT,
                "num_ctx": NUM_CTX,
            },
        }

        try:

            data = json.dumps(
                payload
            ).encode("utf-8")

            request = urllib.request.Request(
                OLLAMA_URL,
                data=data,
                headers={
                    "Content-Type":
                    "application/json"
                },
                method="POST",
            )

            answer = ""

            last_update = time.monotonic()

            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:

                for line in response:

                    if not line:
                        continue

                    try:

                        item = json.loads(
                            line.decode(
                                "utf-8"
                            )
                        )

                    except json.JSONDecodeError:

                        continue

                    chunk = item.get(
                        "response",
                        "",
                    )

                    if not chunk:
                        continue

                    answer += chunk

                    now = time.monotonic()

                    if (
                        now - last_update
                        >= UI_UPDATE_INTERVAL
                    ):

                        AppHelper.callAfter(
                            self.show_stream,
                            answer,
                        )

                        last_update = now

            AppHelper.callAfter(
                self.finish_answer,
                answer,
            )

        except urllib.error.URLError as error:

            AppHelper.callAfter(
                self.show_error,
                (
                    "Unable to connect to Ollama.\n\n"
                    "Run:\n\n"
                    "ollama serve\n\n"
                    f"Error: {error}"
                ),
            )

        except Exception as error:

            AppHelper.callAfter(
                self.show_error,
                f"Error: {error}",
            )

    # ========================================================
    # STREAM
    # ========================================================

    @objc.python_method
    def _set_response_text_white(self):
        """Keep all normal AI response text white, regardless of macOS appearance."""
        if not hasattr(self, "response_view"):
            return

        white = NSColor.colorWithCalibratedWhite_alpha_(0.97, 1.0)
        self.response_view.setTextColor_(white)

        # NSTextView can retain an attributed-string foreground color after
        # setString_. Re-apply white to the complete plain-text content.
        storage = self.response_view.textStorage()
        if storage is not None and storage.length() > 0:
            storage.addAttribute_value_range_(
                "NSForegroundColor",
                white,
                (0, storage.length()),
            )

    @objc.python_method
    def show_stream(
        self,
        answer,
    ):

        visible = re.sub(
            r"```.*",
            "",
            answer,
            flags=re.DOTALL,
        )

        self.response_view.setString_(
            visible
        )
        self._set_response_text_white()

        # Keep streaming clean and minimal while the card grows.
        self.response_view.setDrawsBackground_(False)

        # First streamed content switches from compact prompt-only mode
        # to the response layout, then grows as more text arrives.
        self.grow_response_window()

        self.status.setStringValue_(
            "● GENERATING..."
        )

    @objc.python_method
    def apply_response_code_colors(self, content):
        """Render normal text normally and give fenced code a coding-editor look."""
        if not hasattr(self, "response_view"):
            return

        content = content or ""

        normal_font = NSFont.systemFontOfSize_(14.0)
        code_font = NSFont.fontWithName_size_("Menlo", 13.0)
        if code_font is None:
            code_font = NSFont.monospacedSystemFontOfSize_weight_(13.0, 0.0)

        normal_color = NSColor.colorWithCalibratedWhite_alpha_(
            0.97, 1.0
        )
        code_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.86, 0.90, 0.96, 1.0
        )
        keyword_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.72, 0.48, 0.96, 1.0
        )
        string_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.20, 0.65, 0.42, 1.0
        )
        comment_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.38, 0.45, 0.52, 1.0
        )
        number_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.85, 0.52, 0.20, 1.0
        )
        function_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.16, 0.48, 0.78, 1.0
        )

        result = NSMutableAttributedString.alloc().init()

        # Preserve normal answer text and style fenced code separately.
        pattern = re.compile(r"```(?:[A-Za-z0-9_+#.-]+)?\s*\n?(.*?)```", re.DOTALL)
        pos = 0

        for match in pattern.finditer(content):
            before = content[pos:match.start()]
            if before:
                result.appendAttributedString_(
                    NSMutableAttributedString.alloc().initWithString_attributes_(
                        before,
                        {
                            "NSFont": normal_font,
                            "NSForegroundColor": normal_color,
                        },
                    )
                )

            code = match.group(1).strip("\n")
            code_attrs = NSMutableAttributedString.alloc().initWithString_attributes_(
                code,
                {
                    "NSFont": code_font,
                    "NSForegroundColor": code_color,
                    "NSBackgroundColor": NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        0.07, 0.08, 0.11, 1.0
                    ),
                },
            )

            # Syntax colors for common Python/JSON/SQL/shell constructs.
            for m in re.finditer(
                r"\b(?:def|class|return|import|from|if|else|elif|for|while|in|"
                r"try|except|finally|with|as|async|await|True|False|None|"
                r"and|or|not|is|raise|yield|lambda|SELECT|FROM|WHERE|INSERT|"
                r"UPDATE|DELETE|CREATE|curl|docker)\b",
                code,
                re.IGNORECASE,
            ):
                code_attrs.addAttribute_value_range_(
                    "NSForegroundColor",
                    keyword_color,
                    (m.start(), m.end() - m.start()),
                )

            for m in re.finditer(r"(['\"]{1,3})(?:(?!\1).)*?\1", code, re.DOTALL):
                code_attrs.addAttribute_value_range_(
                    "NSForegroundColor",
                    string_color,
                    (m.start(), m.end() - m.start()),
                )

            for m in re.finditer(r"(?m)#.*$|--.*$", code):
                code_attrs.addAttribute_value_range_(
                    "NSForegroundColor",
                    comment_color,
                    (m.start(), m.end() - m.start()),
                )

            for m in re.finditer(r"\b\d+(?:\.\d+)?\b", code):
                code_attrs.addAttribute_value_range_(
                    "NSForegroundColor",
                    number_color,
                    (m.start(), m.end() - m.start()),
                )

            for m in re.finditer(r"\b[A-Za-z_]\w*(?=\s*\()", code):
                code_attrs.addAttribute_value_range_(
                    "NSForegroundColor",
                    function_color,
                    (m.start(), m.end() - m.start()),
                )

            result.appendAttributedString_(code_attrs)
            pos = match.end()

        remaining = content[pos:]
        if remaining:
            result.appendAttributedString_(
                NSMutableAttributedString.alloc().initWithString_attributes_(
                    remaining,
                    {
                        "NSFont": normal_font,
                        "NSForegroundColor": normal_color,
                    },
                )
            )

        self.response_view.textStorage().setAttributedString_(result)

    @objc.python_method
    def grow_response_window(self):
        """Grow the OUTER window together with the AI response card."""
        if not hasattr(self, "response_view") or not hasattr(self, "response_scroll"):
            return

        try:
            text = self.response_view.string() or ""
            width = max(
                300.0,
                self.response_scroll.frame().size.width
                - (self.content_side_padding * 2),
            )

            font = self.response_view.font() or NSFont.systemFontOfSize_(16.0)
            storage = NSTextStorage.alloc().initWithString_attributes_(
                text, {"NSFont": font}
            )
            container = NSTextContainer.alloc().initWithContainerSize_(
                NSMakeSize(width, 100000.0)
            )
            container.setLineFragmentPadding_(0.0)

            layout = NSLayoutManager.alloc().init()
            layout.addTextContainer_(container)
            storage.addLayoutManager_(layout)

            used = layout.usedRectForTextContainer_(container)
            text_height = max(1.0, used.size.height)

            card_height = min(
                self.response_card_max_height,
                max(self.response_card_min_height, text_height + 32.0),
            )

            # The complete window height follows the response card height.
            chrome_height = 190.0
            desired_height = min(
                self.response_window_max_height,
                max(self.response_window_min_height, chrome_height + card_height),
            )

            frame = self.window.frame()
            current_height = frame.size.height

            # Keep the top of the window in the same place while it grows.
            new_frame = NSMakeRect(
                frame.origin.x,
                frame.origin.y - (desired_height - current_height),
                frame.size.width,
                desired_height,
            )

            self.window.setFrame_display_animate_(new_frame, True, False)

            # Re-layout the entire window so the response card fills the
            # newly-created space instead of growing inside a fixed window.
            self._layout_content(new_frame.size.width, new_frame.size.height)

        except Exception as exc:
            print(f"Response window resize error: {exc}")

    @objc.python_method
    def finish_answer(
        self,
        answer,
    ):

        code_matches = re.findall(
            r"```(?:[\w#+.-]+)?\s*(.*?)```",
            answer,
            re.DOTALL,
        )

        if code_matches:

            code = "\n\n".join(
                item.strip()
                for item in code_matches
                if item.strip()
            )

            clean_answer = re.sub(
                r"```(?:[\w#+.-]+)?\s*(.*?)```",
                "",
                answer,
                flags=re.DOTALL,
            ).strip()

            self.code_view.setString_(
                code
            )

        else:

            clean_answer = answer.strip()

            self.code_view.setString_(
                "No code required."
            )

        # Keep the complete final answer, including code, in the main response.
        self.apply_response_code_colors(clean_answer)
        self._set_response_text_white()

        self.has_answer = True
        self.grow_response_window()

        if getattr(self, "code_only_request", False):
            # Code-only mode: one clean code panel, no explanatory response/code split.
            # Keep everything in the single main response panel.
            self.code_header.setHidden_(True)
            self.code_scroll.setHidden_(True)
            self.nav_code_button.setHidden_(True)
            self.nav_definition_button.setHidden_(True)
            self.nav_view_button.setHidden_(True)
        else:
            self.show_answer_controls()

        self.ask_button.setEnabled_(
            True
        )

        self.mic_button.setEnabled_(
            True
        )

        self.set_status(
            "● LOCAL AI",
            NSColor.systemGreenColor(),
        )

    # ========================================================
    # ERROR
    # ========================================================

    @objc.python_method
    def show_error(
        self,
        message,
    ):

        self.recording = False

        self._set_nav_icon(self.mic_button, "mic.fill", "Microphone")

        self.response_view.setString_(
            message
        )

        self.code_view.setString_(
            "No code available."
        )

        self.ask_button.setEnabled_(
            True
        )

        self.mic_button.setEnabled_(
            True
        )

        if hasattr(self, "screenshot_button"):
            self.screenshot_button.setEnabled_(True)

        self.screenshot_processing = False

        try:
            self.window.makeKeyAndOrderFront_(None)
        except Exception:
            pass

        self.set_status(
            "● ERROR",
            NSColor.systemRedColor(),
        )

    # ========================================================
    # DEFINITION VIEW
    # ========================================================

    def showDefinition_(self, sender):
        current = self.response_view.string() or ""

        if self.definition_only:
            if self._definition_backup is not None:
                self.response_view.setString_(
                    self._definition_backup
                )
            self.definition_only = False
            return

        if not current.strip():
            return

        match = re.search(
            r"(?is)(?:^|\n)\s*Definition\s*:\s*(.*?)(?=\n\s*(?:Key Points|Suggested Response|Explanation)\s*:|$)",
            current,
        )

        if match:
            definition = match.group(1).strip()
            self._definition_backup = current
            self.response_view.setString_(
                "Definition:\n\n" + definition
            )
            self.definition_only = True
            self.set_status(
                "● DEFINITION",
                NSColor.systemGreenColor(),
            )
        else:
            self.set_status(
                "● NO DEFINITION",
                NSColor.systemOrangeColor(),
            )

    # ========================================================
    # VIEW ONLY
    # ========================================================

    def toggleViewOnly_(self, sender):
        self.view_only = not self.view_only

        if self.view_only:
            self.ask_label.setHidden_(True)
            self.prompt.setHidden_(True)
            self.response_header.setFrame_(
                NSMakeRect(25, 650, 695, 32)
            )
            self.response_scroll.setFrame_(
                NSMakeRect(25, 425, 695, 220)
            )
            self.code_header.setHidden_(True)
            self.code_scroll.setHidden_(True)

            self.nav_view_button.setContentTintColor_(
                NSColor.systemGreenColor()
            )
            self.set_status(
                "● VIEW ONLY",
                NSColor.systemGreenColor(),
            )
        else:
            self.ask_label.setHidden_(False)
            self.prompt.setHidden_(False)
            self.response_header.setFrame_(
                NSMakeRect(25, 575, 695, 32)
            )
            self.response_scroll.setFrame_(
                NSMakeRect(25, 345, 695, 225)
            )
            self.code_header.setHidden_(False)
            self.code_scroll.setHidden_(not self.code_expanded)

            self.nav_view_button.setContentTintColor_(
                NSColor.blackColor()
            )
            self.set_status(
                "● READY",
                NSColor.systemGreenColor(),
            )

    # ========================================================
    # RESPONSE COLLAPSE
    # ========================================================

    def toggleResponse_(
        self,
        sender,
    ):

        self.response_expanded = (
            not self.response_expanded
        )

        if self.response_expanded:

            self.response_scroll.setHidden_(
                False
            )

            self.response_header.setTitle_(
                "AI RESPONSE"
            )

            self.resize_response(
                True
            )

        else:

            self.response_scroll.setHidden_(
                True
            )

            self.response_header.setTitle_(
                "AI RESPONSE"
            )

            self.resize_response(
                False
            )

    # ========================================================
    # CODE COLLAPSE
    # ========================================================

    def toggleCode_(
        self,
        sender,
    ):

        self.code_expanded = (
            not self.code_expanded
        )

        self.code_header.setHidden_(False)

        if self.code_expanded:
            self.code_scroll.setHidden_(False)
        else:
            self.code_scroll.setHidden_(True)

        self.code_header.setTitle_("CODE")
        self.layout_main_ui()

    # ========================================================
    # RESPONSE RESIZE
    # ========================================================

    @objc.python_method
    def resize_response(
        self,
        expanded,
    ):

        if self.view_only:
            return

        if expanded:

            self.response_scroll.setFrame_(
                NSMakeRect(
                    25,
                    345,
                    695,
                    225,
                )
            )

            self.code_header.setFrame_(
                NSMakeRect(
                    25,
                    305,
                    695,
                    32,
                )
            )

        else:

            self.response_scroll.setFrame_(
                NSMakeRect(
                    25,
                    345,
                    695,
                    1,
                )
            )

            self.code_header.setFrame_(
                NSMakeRect(
                    25,
                    525,
                    695,
                    32,
                )
            )

    # ========================================================
    # MORE
    # ========================================================

    def more_(
        self,
        sender,
    ):

        question = (
            self.prompt.stringValue()
            .strip()
        )

        if not question:
            return

        previous = (
            self.response_view.string()
        )

        prompt = f"""Question:
{question}

Previous answer:
{previous}

Give only additional useful information.

Rules:
- Do not repeat the previous answer.
- Be concise.
- Maximum 60 words.
- If code is required, use one markdown code block.
"""

        self.status.setStringValue_(
            "● THINKING..."
        )

        thread = threading.Thread(
            target=self.ask_more_ollama,
            args=(prompt,),
            daemon=True,
        )

        thread.start()

    # ========================================================
    # MORE OLLAMA
    # ========================================================

    @objc.python_method
    def ask_more_ollama(
        self,
        prompt,
    ):

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "keep_alive": -1,

            "options": {
                "temperature": 0.1,
                "num_predict": 80,
                "num_ctx": 768,
            },
        }

        try:

            data = json.dumps(
                payload
            ).encode("utf-8")

            request = urllib.request.Request(
                OLLAMA_URL,
                data=data,
                headers={
                    "Content-Type":
                    "application/json"
                },
                method="POST",
            )

            answer = ""

            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:

                for line in response:

                    if not line:
                        continue

                    try:

                        item = json.loads(
                            line.decode(
                                "utf-8"
                            )
                        )

                    except json.JSONDecodeError:

                        continue

                    answer += item.get(
                        "response",
                        "",
                    )

            AppHelper.callAfter(
                self.append_more,
                answer,
            )

        except Exception as error:

            AppHelper.callAfter(
                self.show_error,
                f"Error: {error}",
            )

    # ========================================================
    # APPEND MORE
    # ========================================================

    @objc.python_method
    def append_more(
        self,
        answer,
    ):

        code_matches = re.findall(
            r"```(?:[\w#+.-]+)?\s*(.*?)```",
            answer,
            re.DOTALL,
        )

        if code_matches:

            code = "\n\n".join(
                item.strip()
                for item in code_matches
                if item.strip()
            )

            self.code_view.setString_(
                code
            )

            answer = re.sub(
                r"```(?:[\w#+.-]+)?\s*(.*?)```",
                "",
                answer,
                flags=re.DOTALL,
            ).strip()

        old = self.response_view.string()

        self.response_view.setString_(
            old
            + "\n\nAdditional Information:\n\n"
            + answer.strip()
        )

        self.set_status(
            "● LOCAL AI",
            NSColor.systemGreenColor(),
        )

    # ========================================================
    # CODE POPUP
    # ========================================================

    def openPopup_(
        self,
        sender,
    ):

        if self.popup_window is not None:

            self.popup_window.makeKeyAndOrderFront_(
                None
            )

            self.popup_window.makeFirstResponder_(
                self.popup_code_view
            )

            return

        self.create_popup()

    # ========================================================
    # CREATE POPUP
    # ========================================================

    @objc.python_method
    def create_popup(self):

        screen = (
            self.window
            .screen()
            .visibleFrame()
        )

        width = 850
        height = 650

        x = (
            screen.origin.x
            + (
                screen.size.width
                - width
            ) / 2
        )

        y = (
            screen.origin.y
            + (
                screen.size.height
                - height
            ) / 2
        )

        frame = NSMakeRect(
            x,
            y,
            width,
            height,
        )

        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskResizable
        )

        self.popup_window = (
            NSWindow.alloc()
            .initWithContentRect_styleMask_backing_defer_(
                frame,
                style,
                NSBackingStoreBuffered,
                False,
            )
        )

        self.popup_window.setTitle_(
            "Code"
        )

        self.popup_window.setLevel_(
            NSFloatingWindowLevel
        )

        self.popup_window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        self.popup_window.setSharingType_(
            NSWindowSharingNone
        )

        self.popup_window.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.055, 0.065, 0.085, 1.0
            )
        )

        content = (
            self.popup_window.contentView()
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        title = (
            NSTextField.alloc()
            .initWithFrame_(
                NSMakeRect(
                    20,
                    height - 55,
                    600,
                    30,
                )
            )
        )

        title.setStringValue_(
            "CODE"
        )

        title.setFont_(
            NSFont.boldSystemFontOfSize_(18)
        )

        title.setTextColor_(
            NSColor.blackColor()
        )

        title.setDrawsBackground_(False)
        title.setBezeled_(False)
        title.setEditable_(False)

        content.addSubview_(
            title
        )

        # ----------------------------------------------------
        # HELP
        # ----------------------------------------------------

        help_text = (
            NSTextField.alloc()
            .initWithFrame_(
                NSMakeRect(
                    20,
                    15,
                    800,
                    25,
                )
            )
        )

        help_text.setStringValue_(
            "↑↓ Scroll   ←→ Move   "
            "⌘↑↓ Move vertically   Esc Close"
        )

        help_text.setFont_(
            NSFont.systemFontOfSize_(11)
        )

        help_text.setTextColor_(
            NSColor.lightGrayColor()
        )

        help_text.setDrawsBackground_(False)
        help_text.setBezeled_(False)
        help_text.setEditable_(False)

        content.addSubview_(
            help_text
        )

        # ----------------------------------------------------
        # SCROLL
        # ----------------------------------------------------

        self.popup_scroll = (
            NSScrollView.alloc()
            .initWithFrame_(
                NSMakeRect(
                    20,
                    55,
                    width - 40,
                    height - 125,
                )
            )
        )

        self.popup_scroll.setHasVerticalScroller_(
            True
        )

        self.popup_scroll.setHasHorizontalScroller_(
            True
        )

        # ----------------------------------------------------
        # CODE VIEW
        # ----------------------------------------------------

        self.popup_code_view = (
            CodeTextView.alloc()
            .initWithFrame_(
                NSMakeRect(
                    0,
                    0,
                    width - 60,
                    height - 145,
                )
            )
        )

        self.popup_code_view.setEditable_(
            False
        )

        self.popup_code_view.setSelectable_(
            True
        )

        self.popup_code_view.setFont_(
            NSFont.monospacedSystemFontOfSize_weight_(
                14,
                0.0,
            )
        )

        self.popup_code_view.setTextColor_(
            NSColor.colorWithCalibratedWhite_alpha_(
                0.95,
                1.0,
            )
        )

        self.popup_code_view.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.055, 0.065, 0.085, 1.0
            )
        )

        self.popup_code_view.setRichText_(
            False
        )

        self.popup_code_view.setString_(
            self.code_view.string()
        )

        self.popup_code_view.controller = self

        self.popup_scroll.setDocumentView_(
            self.popup_code_view
        )

        content.addSubview_(
            self.popup_scroll
        )

        self.popup_window.makeKeyAndOrderFront_(
            None
        )

        self.popup_window.makeFirstResponder_(
            self.popup_code_view
        )

        NSApplication.sharedApplication().activateIgnoringOtherApps_(
            True
        )

    # ========================================================
    # POPUP SCROLL
    # ========================================================

    @objc.python_method
    def scroll_popup(
        self,
        amount,
    ):

        if self.popup_scroll is None:
            return

        clip = (
            self.popup_scroll.contentView()
        )

        bounds = clip.bounds()

        bounds.origin.y += amount

        document = (
            self.popup_code_view
        )

        max_y = max(
            0,
            document.frame().size.height
            - clip.bounds().size.height,
        )

        bounds.origin.y = max(
            0,
            min(
                bounds.origin.y,
                max_y,
            ),
        )

        clip.setBoundsOrigin_(
            bounds.origin
        )

    # ========================================================
    # MOVE POPUP
    # ========================================================

    @objc.python_method
    def move_popup(
        self,
        dx,
        dy,
    ):

        if self.popup_window is None:
            return

        frame = (
            self.popup_window.frame()
        )

        frame.origin.x += dx
        frame.origin.y += dy

        self.popup_window.setFrame_display_(
            frame,
            True,
        )

    # ========================================================
    # CLOSE POPUP
    # ========================================================

    @objc.python_method
    def close_popup(self):

        if self.popup_window is not None:

            self.popup_window.orderOut_(
                None
            )

            self.popup_window = None
            self.popup_code_view = None
            self.popup_scroll = None


# ============================================================
# START
# ============================================================

app = NSApplication.sharedApplication()

delegate = AppDelegate.alloc().init()

app.setDelegate_(
    delegate
)

app.setActivationPolicy_(0)
app.activateIgnoringOtherApps_(True)

app.run()