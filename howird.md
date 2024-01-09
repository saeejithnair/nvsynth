# notes

- bring up with:
```
docker compose up -d
docker compose exec nvsynth bash
```

- within container run:

```
/isaac-sim/runheadless.native.sh
```
- other options include: `runheadless.webrtc.sh`, `runheadless.websocket.h264.sh`, `runheadless.websocket.sh`
    - i cannot really get any of these to work, the program runs, but when i try to connect the client it does nothing
    - each of them use a different set of ports as described in the docker-compose file
        - i checked the ports using: `lsof -nP -iTCP -sTCP:LISTEN`

- the only one that has some promise to be working is `runheadless.native.sh`
- after running `/isaac-sim/runheadless.native.sh` in one shell, in a different one look at the logs with:
```
tail -f /root/.nvidia-omniverse/logs/Kit/Isaac-Sim/2023.1/kit_YYYYMMDD_HHMMSS.log
```
- when we connect our client, we get a black screen for a couple of secs and then this in the logs:
```
2024-01-09 06:58:35 [19,908ms] [Info] [omni.kit.app.plugin] [19.936s] Isaac Sim Headless Native App is loaded.
2024-01-09 06:58:46 [30,321ms] [Info] [carb.livestream.plugin] Stream server: connected stream 0x7faad402d760 on connection 0x7faa54026280
2024-01-09 06:58:56 [40,533ms] [Info] [carb.livestream.plugin] Stream Server: stream0 0x7faa54026280 (type 1) stopped
2024-01-09 06:58:56 [40,533ms] [Info] [carb.livestream.plugin] Stream Server: stream1 0x7faa540397f0 (type 4) stopped
2024-01-09 06:58:56 [40,533ms] [Info] [carb.livestream.plugin] Client (nil) disconnected.
```

- also, in the terminal containing the actual app process, we get:
```
[19.936s] Isaac Sim Headless Native App is loaded.
main: thread_init: already added for thread
main: thread_init: already added for thread
```