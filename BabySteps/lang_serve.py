{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "merhaba dünya\n"
     ]
    },
    {
     "ename": "RuntimeError",
     "evalue": "asyncio.run() cannot be called from a running event loop",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 30\u001b[0m\n\u001b[0;32m     28\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;18m__name__\u001b[39m \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m__main__\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[0;32m     29\u001b[0m     \u001b[38;5;28mprint\u001b[39m(chain\u001b[38;5;241m.\u001b[39minvoke({\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mlanguage\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mTurkish\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtext\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhello world\u001b[39m\u001b[38;5;124m\"\u001b[39m}))\n\u001b[1;32m---> 30\u001b[0m     \u001b[43muvicorn\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mrun\u001b[49m\u001b[43m(\u001b[49m\u001b[43mapp\u001b[49m\u001b[43m,\u001b[49m\u001b[43mhost\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43m0.0.0.0\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43mport\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m8000\u001b[39;49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mc:\\Users\\cengh\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\uvicorn\\main.py:577\u001b[0m, in \u001b[0;36mrun\u001b[1;34m(app, host, port, uds, fd, loop, http, ws, ws_max_size, ws_max_queue, ws_ping_interval, ws_ping_timeout, ws_per_message_deflate, lifespan, interface, reload, reload_dirs, reload_includes, reload_excludes, reload_delay, workers, env_file, log_config, log_level, access_log, proxy_headers, server_header, date_header, forwarded_allow_ips, root_path, limit_concurrency, backlog, limit_max_requests, timeout_keep_alive, timeout_graceful_shutdown, ssl_keyfile, ssl_certfile, ssl_keyfile_password, ssl_version, ssl_cert_reqs, ssl_ca_certs, ssl_ciphers, headers, use_colors, app_dir, factory, h11_max_incomplete_event_size)\u001b[0m\n\u001b[0;32m    575\u001b[0m         Multiprocess(config, target\u001b[38;5;241m=\u001b[39mserver\u001b[38;5;241m.\u001b[39mrun, sockets\u001b[38;5;241m=\u001b[39m[sock])\u001b[38;5;241m.\u001b[39mrun()\n\u001b[0;32m    576\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 577\u001b[0m         \u001b[43mserver\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mrun\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    578\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyboardInterrupt\u001b[39;00m:\n\u001b[0;32m    579\u001b[0m     \u001b[38;5;28;01mpass\u001b[39;00m  \u001b[38;5;66;03m# pragma: full coverage\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\cengh\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\uvicorn\\server.py:65\u001b[0m, in \u001b[0;36mServer.run\u001b[1;34m(self, sockets)\u001b[0m\n\u001b[0;32m     63\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mrun\u001b[39m(\u001b[38;5;28mself\u001b[39m, sockets: \u001b[38;5;28mlist\u001b[39m[socket\u001b[38;5;241m.\u001b[39msocket] \u001b[38;5;241m|\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m     64\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig\u001b[38;5;241m.\u001b[39msetup_event_loop()\n\u001b[1;32m---> 65\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43masyncio\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mrun\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mserve\u001b[49m\u001b[43m(\u001b[49m\u001b[43msockets\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msockets\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mc:\\Users\\cengh\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\asyncio\\runners.py:190\u001b[0m, in \u001b[0;36mrun\u001b[1;34m(main, debug, loop_factory)\u001b[0m\n\u001b[0;32m    161\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"Execute the coroutine and return the result.\u001b[39;00m\n\u001b[0;32m    162\u001b[0m \n\u001b[0;32m    163\u001b[0m \u001b[38;5;124;03mThis function runs the passed coroutine, taking care of\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    186\u001b[0m \u001b[38;5;124;03m    asyncio.run(main())\u001b[39;00m\n\u001b[0;32m    187\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    188\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m events\u001b[38;5;241m.\u001b[39m_get_running_loop() \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m    189\u001b[0m     \u001b[38;5;66;03m# fail fast with short traceback\u001b[39;00m\n\u001b[1;32m--> 190\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mRuntimeError\u001b[39;00m(\n\u001b[0;32m    191\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124masyncio.run() cannot be called from a running event loop\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m    193\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m Runner(debug\u001b[38;5;241m=\u001b[39mdebug, loop_factory\u001b[38;5;241m=\u001b[39mloop_factory) \u001b[38;5;28;01mas\u001b[39;00m runner:\n\u001b[0;32m    194\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m runner\u001b[38;5;241m.\u001b[39mrun(main)\n",
      "\u001b[1;31mRuntimeError\u001b[0m: asyncio.run() cannot be called from a running event loop"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
