# 시스템 이상 탐지 및 Cryptojacking Malware 대응 실습 정리

---

## 1. 개요 (Overview)

실제 현업에서 경험한 시스템 이상 탐지 사례를 바탕으로, 비정상적인 CPU 점유 및 악성 프로세스 확인, 그리고 그에 따른 보안 조치 사항을 정리한 문서입니다.

이 사례에서는 외부에서 침투한 후 내부 시스템에 확산되는 **Cryptojacking malware**를 탐지하고 제거하는 과정을 설명합니다.

---

## 2. 이상 징후 탐지 및 초기 분석

### ▸ 이상 증상

* 시스템의 CPU 사용량이 비정상적으로 높음
* `top` 명령어로 확인한 결과, `sh` 프로세스가 CPU를 199% 사용 중이며, 장시간 실행됨

```bash
top - 04:47:47 up 308 days, 22:06,  3 users,  load average: 4.01, 4.04, 4.08
Tasks: 312 total,   1 running, 311 sleeping,   0 stopped,   0 zombie
%Cpu(s): 50.3 us,  0.8 sy,  0.0 ni, 48.8 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem :  7910632 total,   192236 free,  5199368 used,  2519028 buff/cache
KiB Swap:  6713340 total,  6710772 free,     2568 used.  2032884 avail Mem

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
 8440 {USER}     20   0 2445304   2.3g      8 S 199.0 30.4  11249:02 sh
11182 {USER}     20   0  801708 100664   9372 S   2.0  1.3   4030:43 gsd-color
```

### ▸ 비정상 프로세스 식별

```bash
ps -ef | grep 8440
{USER}    8440 1  0  2024 ?        2-19:10:43 /usr/bin/sshd
```

* 해당 바이너리는 gnome 관련 정상 프로세스로 위장되었으나, 자원 점유 및 실행 시간이 과도하여 악성 코드로 의심됨
* 파일 확인 결과:

```bash
ls -l /usr/bin/sshd
ls: cannot access /usr/bin/sshd: No such file or directory
```

* 해당 실행파일은 악성 프로세스가 임시로 파일을 생성 해 실행한 뒤 삭제한 경우 확인되지 않을 수 있다.
* 이러한 경우 비 정상 프로세스를 kill 했을 때 다시 프로세스가 살아나는 경우를 역추적 해 근본 Malware를 제거해야 한다.



---

## 3. 사용자 확인 및 내부 접근 분석

### ▸ 현재 로그인 사용자 확인

```bash
w
who
```

```bash
USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
dbsec    :0       :0               22Jul24 ?xdm?  12days  1:09  /usr/libexec/gnome-session-binary --session gnome-classic
dbsec    pts/0    10.120.130.11    04:47    0.00s  0.13s  0.03s w
dbsec    pts/2    :0               23Jul24 307days  0.16s  0.67s /usr/libexec/gnome-terminal-server
```

* 내부망 IP `10.120.130.11`에서의 SSH 접속 확인

### ▸ 네트워크 위치 및 확산 경로 분석

* 시스템은 **내부망**에 위치하며, 외부에서 직접 접근은 불가능하지만, 내부에서 외부로의 **outbound는 허용**된 상태
* **외부망 접속이 허용된 내부 장비로부터 침투 후 lateral movement가 발생한 것으로 추정**

---

## 4. 감염 경로 차단 및 보안 조치

### ▸ 취약 계정 보안 강화

* 시스템 내 등록된 계정 중 비밀번호가 취약한 계정을 대상으로 모두 변경

### ▸ SSH 인증 기반 차단

* `.ssh/authorized_keys`, `.ssh/known_hosts` 등을 점검 및 갱신하여 공개키 기반 자동접속 차단

---

## 5. 악성코드 자동 실행 제거 (스케쥴링 확인)

### 1️⃣ `crontab` 기반 스케쥴 확인

```bash
for user in $(cut -f1 -d: /etc/passwd); do echo "=== $user ==="; crontab -l -u $user 2>/dev/null; done
```

* 결과 예시:

```bash
=== {USER} ===
* * * * * /home/{USER}/updated/upd >/dev/null 2>&1
```

→ `crontab -e`로 해당 항목 제거

### 2️⃣ `systemd` 및 `service` 기반 autorun 제거

```bash
systemctl list-units --type=service | grep {malware name}
ls /etc/init.d | grep {malware name}
```

→ 발견 시 disable 및 파일 제거

---

## 6. 파일 속성 보호 우회 및 삭제

### ▸ 삭제 실패 메시지

```bash
rm: cannot remove 'updated/xxxx': Operation not permitted
```

### ▸ `lsattr`로 확인

```bash
lsattr *
----ia---------- config.json
...
```

### ▸ 속성 해제 및 삭제

```bash
chattr -i -a *
rm -rf *
```

---

## 7. 악성코드 위장 프로세스 및 SSH 탐지

### ▸ `/usr/bin/sshd` vs `/usr/sbin/sshd`

* `/usr/sbin/sshd`: 정상 데몬 (root 실행)
* `/usr/bin/sshd`: dbsec 사용자에 의해 수동 실행, CPU 99%, PID 31137 → **악성 코드로 의심**

```bash
ps -ef | grep sshd
dbsec    31137     1 99 05:09 ?        00:06:41 /usr/bin/sshd
```

### ▸ `ssh-agent` 위장 탐지

```bash
ls /usr/bin | grep ssh
```

* `ssh-agent`, `slogin` 등이 예상치 못한 사용자 실행 여부 확인 필요

---

## 8. 보안 위협 총평 및 권고 사항

* 본 사례는 외부망과 연계된 장비를 통해 내부망 전체로 악성코드가 확산된 전형적인 **Cryptojacking campaign** 양상
* 단일 시스템 뿐 아니라 **주변 개발 서버로 확산**되며, 동일한 계정 정보 사용으로 인한 보안 취약점이 핵심 확산 경로였음
* `.ssh` 인증 정보 탈취와 `crontab`, `systemd`, `attr` 보호 기법이 모두 활용되었으며, `sshd` 및 `sh` 프로세스가 위장됨

### 🔐 권고 사항

* **비밀번호 정책 강화**: root 및 일반 사용자 대상 전수 변경
* **인증서 기반 SSH 접속**: password 기반 인증 제거
* **IP 기반 접근제어**: 외부망/내부망 구분된 ACL 적용
* **정기적 스케쥴링/파일 속성 점검**: `lsattr`, `chattr`, `cron`, `systemctl` 모니터링

---

## History

작성일: `2025-05-27`
