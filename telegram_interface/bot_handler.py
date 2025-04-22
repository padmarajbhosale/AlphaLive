# GainCraft/telegram_interface/bot_handler.py
# Handles Telegram bot commands and message scheduling.
# FIXED: Added missing Optional import from typing
# FIXED: /daily_report command (removed DB), /status command (direct MT5 check).

import logging
import sys
import os
import threading
import asyncio
import traceback
import datetime
import MetaTrader5 as mt5
import pandas as pd  # Needed for status command formatting
from typing import Optional  # <<< THIS WAS MISSING

try:
    from telegram import Update, Bot
    from telegram.constants import ParseMode
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters as Filters
    from telegram.error import TelegramError
except ImportError:
    print("ERROR: python-telegram-bot library not found. Please install it (`pip install python-telegram-bot --upgrade`)")
    sys.exit(1)


# Add project root to path relative to this file (telegram_interface folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from config.config_loader import get_config
    from shared_state import get_state, set_state, bot_state, state_lock
    # Database imports removed
except ImportError as e:
    print(f"FATAL ERROR: Import failed in bot_handler: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)
if not logger.hasHandlers():  # Add basic handler if needed
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.warning("Basic logging handler added in bot_handler.")

# --- Async Helper to Send Message ---
async def send_telegram_message_async(bot: Bot, chat_id: int, message_text: str):
    """Sends message asynchronously."""
    if not bot:
        logger.error("Bot instance None in send_async.")
        return
    if not chat_id:
        logger.error("Invalid chat_id in send_async.")
        return
    try:
        # Split long messages if necessary (Telegram limit is 4096 chars)
        max_len = 4090
        if len(message_text) > max_len:
            for i in range(0, len(message_text), max_len):
                await bot.send_message(chat_id=chat_id, text=message_text[i:i+max_len], parse_mode=ParseMode.MARKDOWN)
                await asyncio.sleep(0.5)  # Small delay between parts
        else:
            await bot.send_message(chat_id=chat_id, text=message_text, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"Msg sent via bot to {chat_id}.")
    except TelegramError as e:
        logger.error(f"TG API error sending to {chat_id}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error send_async to {chat_id}: {e}")

# --- Sync Function to Schedule Sending ---
def schedule_telegram_message(chat_id_str: Optional[str], message_text: str):
    """Schedules the async send function to run on the bot's event loop."""
    logger.debug(f"schedule_telegram_message called from thread: {threading.current_thread().name}")
    loop = None
    bot_instance = None
    if not chat_id_str:
        logger.error("Cannot schedule msg: chat_id not configured in .env")
        return
    try:
        chat_id = int(chat_id_str)  # Convert chat_id from config to int
    except (ValueError, TypeError):
        logger.error(f"Cannot schedule msg: Invalid chat_id format '{chat_id_str}'")
        return

    with state_lock:
        loop = bot_state.get("bot_event_loop")
        bot_instance = bot_state.get("telegram_bot_instance")
    if not loop:
        logger.error("Cannot schedule msg: Bot loop not found.")
        return
    if not bot_instance:
        logger.error("Cannot schedule msg: Bot instance not found.")
        return

    try:
        # Ensure the loop is running before scheduling
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(send_telegram_message_async(bot_instance, chat_id, message_text), loop)
            logger.info(f"Scheduled msg send to {chat_id}.")
        else:
            logger.error("Cannot schedule msg: Bot event loop is not running.")
    except Exception as e:
        logger.exception(f"Failed schedule msg send: {e}")

# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logger.info(f"Handler: /start from {user.username}")
    reply_text = f"Hi {user.mention_html()}! GainCraft Bot active."
    await update.message.reply_html(reply_text)
    # Optionally send status immediately after start
    await status_command(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Handler: /help from {update.effective_user.username}")
    help_text = (
        "GainCraft Bot Commands:\n"
        "/start - Check bot status\n"
        "/help - Show this help message\n"
        "/status - Get current trading status & MT5 info\n"
        "/pause - Pause placing new trades\n"
        "/resume - Resume placing new trades\n"
        "/daily_report - (WIP) Show today's trade summary\n"
        "/close_all - Request engine to close all open trades"
    )
    await update.message.reply_text(help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fetches and displays current bot and MT5 status."""
    logger.info(f"Handler: /status from {update.effective_user.username}")
    is_running = get_state("is_running", True)  # Assume running if thread active
    is_paused = get_state("is_paused", False)
    last_err = get_state("last_error")
    status_lines = []

    if is_running:
        status_lines.append(f"*Engine State:* {'PAUSED (New trades disabled)' if is_paused else 'RUNNING'}")
    else:
        status_lines.append("*Engine State:* STOPPED (or starting)")

    # --- Check MT5 Status Directly ---
    mt5_conn_status = "DISCONNECTED"
    balance_str = "N/A"
    equity_str = "N/A"
    pos_count_str = "N/A"
    currency = ""
    positions_details = []
    total_floating_pl = 0.0

    try:
        # Run blocking MT5 calls in a separate thread to avoid blocking asyncio loop
        def get_mt5_status():
            nonlocal mt5_conn_status, balance_str, equity_str, pos_count_str, currency, positions_details, total_floating_pl
            try:
                if mt5.terminal_info():  # Check connection
                    mt5_conn_status = "CONNECTED"
                    account_info = mt5.account_info()
                    positions = mt5.positions_get()  # Get all open positions
                    if account_info:
                        currency = account_info.currency
                        balance_str = f"{account_info.balance:.2f}"
                        equity_str = f"{account_info.equity:.2f}"
                    else:
                        logger.warning("account_info() returned None.")
                        balance_str = "Error"
                        equity_str = "Error"

                    if positions is not None:
                        pos_count_str = str(len(positions))
                        if len(positions) > 0:
                            for pos in positions:
                                pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                                digits = mt5.symbol_info(pos.symbol).digits if mt5.symbol_info(pos.symbol) else 5
                                sl_str = f"SL:{pos.sl:.{digits}f}" if pos.sl > 0 else "SL:N/A"
                                tp_str = f"TP:{pos.tp:.{digits}f}" if pos.tp > 0 else "TP:N/A"
                                positions_details.append(
                                    f"  `{pos.ticket}`: {pos.symbol} {pos_type} {pos.volume} @ {pos.price_open:.{digits}f} ({sl_str} {tp_str}) P/L: {pos.profit:.2f}"
                                )
                                total_floating_pl += pos.profit
                    else:
                        logger.warning("positions_get() returned None.")
                        pos_count_str = "Error"
                else:
                    logger.warning("MT5 terminal_info() returned None. Assuming disconnected.")
                    mt5_conn_status = "DISCONNECTED"
            except Exception as e:
                logger.error(f"Exception fetching MT5 status: {e}")
                mt5_conn_status = "Error Fetching"

        await asyncio.to_thread(get_mt5_status)  # Run the sync MT5 calls in thread
    except Exception as e:
        logger.error(f"Error running get_mt5_status in thread: {e}")
        mt5_conn_status = "Error (Thread)"

    # --- Assemble Status Message ---
    status_lines.append(f"*MT5 Connection:* {mt5_conn_status}")
    if mt5_conn_status == "CONNECTED":
        status_lines.append(f"*Balance:* {balance_str} {currency}")
        status_lines.append(f"*Equity:* {equity_str} {currency}")
        status_lines.append(f"*Open Positions ({pos_count_str}):*")
        if positions_details:
            status_lines.extend(positions_details)
            status_lines.append(f"*Total Floating P/L:* {total_floating_pl:.2f} {currency}")
        elif pos_count_str != "Error":
            status_lines.append("  None")

    if last_err:
        status_lines.append(f"*Last Error:* {last_err}")
    status_message = "\n".join(status_lines)

    try:
        logger.debug("Attempting /status reply...")
        await update.message.reply_text(status_message, parse_mode=ParseMode.MARKDOWN)
        logger.info("/status reply sent.")
    except Exception as e:
        logger.exception(f"Failed /status reply: {e}")

async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Handler: /pause from {update.effective_user.username}")
    set_state("is_paused", True)
    reply_text = "Trading Engine PAUSED (will not place new trades)."
    await update.message.reply_text(reply_text)
    logger.info("Engine state set to PAUSED.")

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Handler: /resume from {update.effective_user.username}")
    set_state("is_paused", False)
    reply_text = "Trading Engine RESUMED."
    await update.message.reply_text(reply_text)
    logger.info("Engine state set to RUNNING.")
    await status_command(update, context)  # Show status after resuming

async def daily_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Placeholder for daily report - reads from CSV eventually"""
    logger.info(f"Handler: /daily_report from {update.effective_user.username}")
    report_text = "Daily report functionality (reading from CSV) is not yet implemented."
    try:
        await update.message.reply_text(report_text)
        logger.info("Sent placeholder daily report message.")
    except Exception as e:
        logger.exception(f"Failed /daily_report reply: {e}")

async def close_all_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logger.info(f"Handler: /close_all from {user.username}")
    set_state("close_all_requested", True)  # Set a flag for the executor to potentially check
    reply_text = "Received /close_all request. Manual closure via MT5 recommended for now."
    await update.message.reply_text(reply_text)
    logger.info("Close all flag SET (manual closure recommended).")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Exception caught by TG handler: {context.error}", exc_info=context.error)
    set_state("last_error", f"TG Error: {context.error}")
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="An internal error occurred processing a command.")
        except Exception:
            logger.error("Failed to send error notification to user.")

# --- Main Bot Setup & Polling Function ---
def run_bot_polling():
    token = get_config('TELEGRAM_BOT_TOKEN')
    if not token or token == 'YOUR_COPIED_TOKEN_HERE':
        logger.error("TG token invalid.")
        return
    logger.info("Setting up Telegram application...")
    application = Application.builder().token(token).build()
    with state_lock:
        bot_state["telegram_bot_instance"] = application.bot
        logger.info("Stored telegram bot instance.")
    # Register Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("pause", pause_command))
    application.add_handler(CommandHandler("resume", resume_command))
    application.add_handler(CommandHandler("daily_report", daily_report_command))
    application.add_handler(CommandHandler("close_all", close_all_command))
    application.add_error_handler(error_handler)
    logger.info("Starting Telegram polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Telegram polling stopped.")

# --- Thread Target Function ---
def run_bot_in_thread():
    """Sets up asyncio loop, runs polling, stores loop/bot in shared state."""
    logger.info("Telegram bot thread started. Setting up asyncio loop...")
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Asyncio loop created/set for Telegram thread.")
        with state_lock:
            bot_state["bot_event_loop"] = loop
        logger.info("Stored bot event loop in shared state.")
        run_bot_polling()  # This blocks until polling stops
    except Exception as e:
        logger.exception(f"Exception in TG thread target: {e}")
        set_state("last_error", f"TG Thread Exception: {e}")
    finally:
        logger.info("Telegram bot thread finishing.")

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("Running bot_handler.py directly for testing...")
    # Example: Start the bot polling directly (will block here)
    # run_bot_polling()
    # Or start in a thread for interactive testing
    # test_thread = threading.Thread(target=run_bot_in_thread, daemon=True)
    # test_thread.start()
    # print("Bot thread started. Keep main thread alive (Ctrl+C to exit)...")
    # while True: time.sleep(1)
    pass