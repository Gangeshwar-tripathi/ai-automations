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
from meeting_assistant import MeetingAssistant
from assistant_functionality import AIService, AudioService

from Cocoa import (
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
        self.compact_window_width = 600
        self.compact_window_height = 320
        self.response_window_min_height = 500
        self.response_window_max_height = 800
        self.response_min_window_height = 320
        self.response_max_window_height = 800
        self.has_answer = False

        self.recording = False
        self.audio_data = None

        self.whisper_model = None
        self.ai_service = AIService(OLLAMA_URL, MODEL, VISION_MODEL)
        self.audio_service = AudioService(WHISPER_MODEL)

        # Meeting mode state. The meeting assistant listens to the
        # configured meeting-audio input and sends detected questions
        # to llama3.2:3b.
        self.meeting_mode = False
        self.screenshot_processing = False
        self.meeting_assistant = MeetingAssistant(
            self.show_meeting_answer
        )

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
            600,
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

        self.window.setTitle_("Private AI Assistant")
        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setDelegate_(self)

        self.window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Keep overlay excluded from supported capture.
        self.window.setSharingType_(NSWindowSharingNone)

        self.window.setBackgroundColor_(
               NSColor.colorWithCalibratedWhite_alpha_(
        0.08,
        0.55,
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
                NSMakeRect(20, 18, 150, 25)
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
        self.nav_mic_button = nav_button(
            250,
            "Microphone",
            "Start or stop microphone recording",
            "toggleRecording:",
        )

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

        self.nav_meeting_button = nav_button(
            690,
            "Meeting",
            "Start or stop meeting listening",
            "toggleMeeting:",
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

        # Keep compatibility with existing meeting/status code.
        self.meeting_button = self.nav_meeting_button
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
        self.ask_label.setTextColor_(NSColor.blackColor())
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
            NSFont.boldSystemFontOfSize_(14)
        )
        self.response_header.setAlignment_(0)
        self.response_header.setBordered_(False)
        self.response_header.setContentTintColor_(
            NSColor.blackColor()
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
                NSMakeRect(25, 345, 695, 225)
            )
        )
        self.response_scroll.setHasVerticalScroller_(True)

        self.response_view = (
            NSTextView.alloc()
            .initWithFrame_(
                NSMakeRect(0, 0, 675, 220)
            )
        )
        self.response_view.setEditable_(False)
        self.response_view.setSelectable_(True)
        self.response_view.setFont_(
            NSFont.systemFontOfSize_(15)
        )
        self.response_view.setTextColor_(NSColor.blackColor())
        self.response_view.setBackgroundColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.97, 1.0)
        )
        self.response_view.setRichText_(False)
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
            NSColor.colorWithCalibratedWhite_alpha_(0.97, 1.0)
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
            "Ask a question by typing or using the microphone, "
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
            max(560.0, size.width),
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
        """Modern compact layout that expands only when an answer exists."""

        if not hasattr(self, "window"):
            return

        bounds = self.window.contentView().bounds()
        w = max(560.0, bounds.size.width)
        h = max(320.0, bounds.size.height)

        pad = 24
        nav_h = 54

        # --------------------------------------------------------
        # NAV BAR
        # --------------------------------------------------------
        self.nav_bar.setFrame_(
            NSMakeRect(
                0,
                h - nav_h,
                w,
                nav_h,
            )
        )

        self.nav_status.setFrame_(
            NSMakeRect(18, 15, 140, 24)
        )

        self.status.setFrame_(
            NSMakeRect(w - 125, 15, 105, 24)
        )

        # Keep navigation controls compact.
        buttons = [
            self.nav_mic_button,
            self.nav_screenshot_button,
            self.nav_code_button,
            self.nav_definition_button,
            self.nav_view_button,
            self.nav_meeting_button,
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
            "Meeting": 78,
        }

        gap = 5
        total_width = sum(
            button_widths.get(str(button.title()), 80)
            for button in visible
        )
        total_width += max(0, len(visible) - 1) * gap

        left_limit = 155
        right_limit = w - 125

        if total_width <= max(0, right_limit - left_limit):
            x = max(
                left_limit,
                (w - total_width) / 2,
            )

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
        prompt_y = h - nav_h - 112

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
                w - 2 * pad,
                40,
            )
        )

        # --------------------------------------------------------
        # RESPONSE
        # --------------------------------------------------------
        if not self.has_answer:
            # Prompt-only compact mode.
            self.response_header.setHidden_(True)
            self.response_scroll.setHidden_(True)
            self.code_header.setHidden_(True)
            self.code_scroll.setHidden_(True)
            return

        self.response_header.setHidden_(False)

        response_header_y = prompt_y - 42

        self.response_header.setFrame_(
            NSMakeRect(
                pad,
                response_header_y,
                w - 2 * pad,
                30,
            )
        )

        # Use all available space below the response header.
        response_y = 28
        response_h = max(
            120,
            response_header_y - response_y - 10,
        )

        self.response_scroll.setFrame_(
            NSMakeRect(
                pad,
                response_y,
                w - 2 * pad,
                response_h,
            )
        )
        self.response_scroll.setHidden_(False)

        # Code remains hidden until code is actually available.
        if self.code_header.isHidden():
            self.code_scroll.setHidden_(True)
            return

        code_header_y = response_y + response_h + 8

        self.code_header.setFrame_(
            NSMakeRect(
                pad,
                code_header_y,
                w - 2 * pad,
                30,
            )
        )

        if self.code_expanded:
            self.code_scroll.setFrame_(
                NSMakeRect(
                    pad,
                    60,
                    w - 2 * pad,
                    max(
                        40,
                        code_header_y - 68,
                    ),
                )
            )
            self.code_scroll.setHidden_(False)
        else:
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
        self.stop_window_arrow_move()

        try:
            if self.meeting_mode:
                self.meeting_mode = False
                self.meeting_assistant.stop()
        except Exception:
            pass
        try:
            if self.recording:
                self.recording = False
                sd.stop()
        except Exception:
            pass

    def closeApp_(self, sender):
        self.stop_window_arrow_move()

        try:
            if self.meeting_mode:
                self.meeting_mode = False
                self.meeting_assistant.stop()
        except Exception:
            pass
        try:
            if self.recording:
                self.recording = False
                sd.stop()
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
    # MEETING MODE
    # ========================================================

    def toggleMeeting_(self, sender):

        if self.meeting_mode:

            self.meeting_mode = False

            self.meeting_assistant.stop()

            self._set_nav_icon(
                self.meeting_button, "headphones", "MEET"
            )

            self.set_status(
                "● LOCAL AI",
                NSColor.systemGreenColor(),
            )

            return

        self.meeting_mode = True

        self._set_nav_icon(
            self.meeting_button, "headphones.circle.fill", "MEET"
        )

        self.set_status(
            "● LISTENING...",
            NSColor.systemRedColor(),
        )

        try:
            self.meeting_assistant.start()
        except Exception as error:
            self.meeting_mode = False
            self._set_nav_icon(
                self.meeting_button, "headphones", "MEET"
            )
            self.show_error(
                f"Unable to start meeting mode.\n\n{error}"
            )

    @objc.python_method
    def show_meeting_answer(
        self,
        question,
        answer,
    ):

        # MeetingAssistant invokes this callback from its worker
        # thread, so update Cocoa UI on the main thread.
        AppHelper.callAfter(
            self._show_meeting_answer_ui,
            question,
            answer,
        )

    @objc.python_method
    def _show_meeting_answer_ui(
        self,
        question,
        answer,
    ):

        self.prompt.setStringValue_(
            question
        )

        self.response_view.setString_(
            "Question:\n\n"
            + question
            + "\n\n"
            + answer.strip()
        )

        self.show_answer_controls()

        self.set_status(
            "● MEETING AI",
            NSColor.systemGreenColor(),
        )

    # ========================================================
    # SCREENSHOT -> LLAVA -> LLAMA
    # ========================================================

    def processScreenshot_(self, sender):
        if self.screenshot_processing:
            return
        self.screenshot_processing = True
        self.screenshot_button.setEnabled_(False)
        self.mic_button.setEnabled_(False)
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
        self.code_view.setString_(screen_context.strip())
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
    def start_recording(self):

        if self.whisper_model is None:

            self.set_status(
                "● VOICE MODEL LOADING...",
                NSColor.systemOrangeColor(),
            )

            return

        self.recording = True

        self.audio_data = []

        self._set_nav_icon(self.mic_button, "stop.fill", "STOP")

        self.status.setStringValue_(
            "● LISTENING..."
        )

        self.status.setTextColor_(
            NSColor.systemRedColor()
        )

        self.response_view.setString_(
            "Listening...\n\nSpeak your question."
        )

        thread = threading.Thread(
            target=self.record_audio,
            daemon=True,
        )

        thread.start()

    # ========================================================
    # RECORD AUDIO
    # ========================================================

    @objc.python_method
    def record_audio(self):

        try:

            recording = sd.rec(
                int(
                    MAX_RECORD_SECONDS
                    * SAMPLE_RATE
                ),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
            )

            # Wait until recording finishes
            # or stop_recording is called.
            while self.recording:

                time.sleep(0.05)

            sd.stop()

            self.audio_data = recording

            AppHelper.callAfter(
                self.transcribe_audio,
            )

        except Exception as error:

            self.recording = False

            AppHelper.callAfter(
                self.show_error,
                (
                    "Microphone error:\n\n"
                    f"{error}"
                ),
            )

    # ========================================================
    # STOP RECORDING
    # ========================================================

    @objc.python_method
    def stop_recording(self):

        self.recording = False

        if self.meeting_mode:
            self.meeting_mode = False
            self.meeting_assistant.stop()
            self._set_nav_icon(
                self.meeting_button, "headphones", "MEET"
            )

        self._set_nav_icon(self.mic_button, "mic.fill", "MIC")

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
                self.show_error("No audio recorded.")
                return
            text = self.audio_service.transcribe(self.audio_data, SAMPLE_RATE)
            AppHelper.callAfter(self.process_transcription, text)
        except Exception as error:
            AppHelper.callAfter(self.show_error, f"Speech recognition error:\n\n{error}")

    # ========================================================
    # PROCESS SPEECH
    # ========================================================

    @objc.python_method
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

        normal_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.12, 0.13, 0.15, 1.0
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
        """Grow the window according to the actual rendered response height."""

        if not hasattr(self, "window") or not hasattr(self, "response_view"):
            return

        if not self.response_expanded or self.view_only:
            return

        self.has_answer = True

        # Lay out first so wrapped lines are included in the measurement.
        self.layout_main_ui()

        layout_manager = self.response_view.layoutManager()
        text_container = self.response_view.textContainer()
        layout_manager.ensureLayoutForTextContainer_(text_container)

        used_rect = layout_manager.usedRectForTextContainer_(text_container)

        # Height actually required by the rendered response.
        required_text_height = max(
            90.0,
            used_rect.size.height + 32.0,
        )

        # Space occupied by navigation, prompt and response header.
        non_response_height = 215.0

        desired_height = max(
            self.response_window_min_height,
            required_text_height + non_response_height,
        )
        desired_height = min(
            desired_height,
            self.response_window_max_height,
        )

        current_frame = self.window.frame()
        current_height = current_frame.size.height

        # Grow only when content needs more room; avoid streaming jitter.
        if desired_height > current_height + 12:
            current_frame.origin.y -= (
                desired_height - current_height
            )
            current_frame.size.height = desired_height

            self.window.setFrame_display_animate_(
                current_frame,
                True,
                False,
            )

            self.layout_main_ui()

        # Keep the newest streamed content visible.
        try:
            length = len(self.response_view.string() or "")
            self.response_view.scrollRangeToVisible_((length, 0))
        except Exception:
            pass

    # ========================================================
    # FINISH
    # ========================================================

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

        self._set_nav_icon(self.mic_button, "mic.fill", "MIC")

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
            NSColor.colorWithCalibratedWhite_alpha_(0.94, 1.0)
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
            NSColor.whiteColor()
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