# RBI ATM Physical and Cyber Security Directives

## Statutory Reference
- **RBI Circular:** RBI/2018-19/166 DCM (Plg) No.2968/10.25.007/2018-19: "Control measures for ATMs - Timeline for implementation"
- **RBI Cyber Security Framework for ATM Switch ASPs:** RBI/2021-22/68 DOS.CO.CSITE.SEC.No.1852/31.01.015/2021-22
- **PCI Security Standards:** PCI PTS (PIN Transaction Security) v5.0 / v6.0 for Encrypting PIN Pads (EPP)

## Key Technical & Regulatory Requirements

### 1. Physical Vault & Grouting Security
- All ATMs must be grouted to a concrete floor/wall structure with high-tensile anchor bolts preventing physical removal.
- Physical vaults must be equipped with vibration, tilt, thermal, and chest-door opening sensors wired into an automated Central Monitoring System (CMS) / e-Surveillance center.
- Alarms must trigger within 5 seconds of physical disturbance with live video streaming automatically initiated to bank security control rooms.

### 2. PIN Pad Security & Key Zeroization
- Encrypting PIN Pads (EPP) must be certified to PCI PTS standards.
- EPP must feature physical tamper-detection mesh circuitry; any drill or displacement attempt must trigger instantaneous cryptographic zeroization (erasure of Terminal Master Keys and Working Keys in <100ms).
- Clear PIN data must never traverse the ATM host CPU or motherboard memory.

### 3. Operating System & Communication Security
- ATMs must run supported, hardened Operating Systems with BIOS passwords and automated USB port disabling.
- Whitelisting software (Application Control) must be enforced to block execution of unapproved binaries or scripts.
- Communications between ATM terminal and core switch must enforce TLS 1.3 or MACing (Message Authentication Code) to prevent 'man-in-the-middle' transaction tampering or black-box cash dispensing attacks.
- Direct Remote Access to ATM OS must be prohibited except through hardened Bastion servers with dual-custody MFA.

### 4. Cash Cassette & Replenishment Controls
- Cash replenishment must implement digital lock-and-key dual custody (Maker-Checker with one-time dynamic electronic keys).
- End-to-end lockable cassette swap systems must ensure cash handlers have zero direct physical access to loose currency notes during transit.
