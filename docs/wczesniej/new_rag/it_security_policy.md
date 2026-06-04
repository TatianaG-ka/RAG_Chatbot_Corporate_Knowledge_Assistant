# Acme Corp — IT Security Policy

**Version 3.7** · Effective date: February 1, 2026 · Owner: Information Security Office

---

## 1. Scope and applicability

This policy applies to **all employees, contractors, interns, and third-party vendors** with access to Acme Corp systems, data, or networks. It governs how access is granted, how credentials are managed, what hardware is permitted, and how security incidents are handled.

The policy applies regardless of physical location — office, home office, travel, or any temporary work location. Contractors are subject to this policy in full from the first day of engagement; their statement of work explicitly references this document.

Violations may result in:

- Immediate revocation of access
- Disciplinary action up to and including termination (employees) or contract termination (contractors)
- Reporting to law enforcement where applicable (e.g., data theft, insider threats)

---

## 2. Account access and credentials

### 2.1 Account provisioning

All new accounts are provisioned through the central identity provider (IdP) within **24 hours** of People Operations confirmation. Direct account creation in individual applications is prohibited — every application must integrate with SSO via SAML 2.0 or OIDC.

Privileged accounts (admin, root, database superuser) require:

- A separate, named account distinct from the user's regular account
- Manager approval AND Information Security Office (ISO) approval
- Justification document signed by both parties
- Mandatory quarterly review of continued need

### 2.2 Password requirements

#### Standard user accounts

- Minimum length: **14 characters**
- Must include at least one uppercase, one lowercase, one number, one special character
- Must not match any of the user's previous 12 passwords
- Must not appear in known compromised password databases (HaveIBeenPwned check at creation)
- Rotation: every **180 days**, or immediately upon suspected compromise

#### Privileged (admin) accounts

- Minimum length: **20 characters**
- Same complexity rules as above
- Rotation: every **90 days** (more frequent than standard accounts)
- Must not be reused across services
- Must be stored exclusively in the company password vault

### 2.3 Multi-factor authentication (MFA)

MFA is **mandatory for all accounts** without exception. Approved MFA methods, in order of preference:

1. Hardware security keys (YubiKey, Google Titan) — required for privileged accounts
2. Authenticator app (Microsoft Authenticator, Google Authenticator, Authy)
3. Push notification to enrolled mobile device

SMS-based MFA is **explicitly prohibited** due to SIM-swap risk. Email-based MFA is permitted only as a temporary fallback during hardware key replacement (max 7 days).

---

## 3. Hardware and device security

### 3.1 Company-issued devices

All work involving Acme Corp data must be performed on company-issued devices. Approved devices:

- Laptops: MacBook Pro / MacBook Air (M-series), ThinkPad (X1 Carbon, T-series, P-series)
- Mobile: iPhone (iOS current -1 major version), Pixel (Android current -1 major version)

Devices must have:

- Full-disk encryption enabled (FileVault on macOS, BitLocker on Windows)
- Endpoint Detection and Response (EDR) agent installed and reporting
- Automatic OS updates within **14 days** of release
- Screen lock with maximum 5-minute inactivity timeout
- Remote wipe capability enrolled before first use

### 3.2 Bring Your Own Device (BYOD)

BYOD is **not permitted** for accessing internal systems, source code, customer data, or financial data. BYOD is permitted only for:

- Reading work email (via mobile email app with MDM container)
- Joining video calls (Zoom, Teams)
- Reading public marketing materials

Personal devices accessing email must enroll in mobile device management (MDM). The MDM scope is limited to the work container — personal data on the device is not accessible to IT.

---

## 4. Network access

### 4.1 VPN

VPN connection is **required** for accessing internal systems from outside the office network. The VPN client (Cisco AnyConnect) is pre-installed on all company devices.

VPN access from the following countries is **blocked** by default and requires explicit ISO approval (typically denied):

- Countries under active EU or US sanctions (current list maintained by Compliance, updated quarterly)
- Countries listed as high-risk by the company's threat intelligence provider

Travel to permitted countries requires no advance approval, but employees must:

- Notify ISO if traveling to any country outside their normal residency for more than 7 days
- Avoid public Wi-Fi for accessing internal systems (use mobile hotspot or VPN-over-cellular)

### 4.2 Office network

Employees connect to the corporate network only via the **acme-corp-secure** SSID. The **acme-guest** SSID is for visitors and personal devices and is segregated from internal systems.

---

## 5. Data classification and handling

### 5.1 Classification levels

| Level | Description | Examples |
| --- | --- | --- |
| **Public** | Information already published or intended for public release | Marketing materials, job postings, press releases |
| **Internal** | Default for all internal documents not otherwise classified | Team strategies, internal wiki, project plans |
| **Confidential** | Restricted to need-to-know basis | Financial forecasts, customer contracts, employee personal data |
| **Restricted** | Strictest control; access logged and audited | Customer payment data, source signing keys, M&A documents |

### 5.2 Handling requirements

- **Confidential** data must not be sent to personal email accounts under any circumstances
- **Restricted** data must not leave approved systems (no copy to local laptop, no screenshots, no printing)
- Customer payment card data is handled exclusively through the PCI-DSS certified payment processor; no employee should ever see full card numbers in any system

### 5.3 Data residency

Customer data is stored in the EU region (Frankfurt) by default. Customers under specific contractual SLAs may have data residency in the US (Virginia) or APAC (Singapore). Cross-region data transfer requires explicit ISO approval and contractual basis.

---

## 6. Security training and incident reporting

### 6.1 Mandatory training

All employees complete:

- **Onboarding security training** — within 5 business days of start date
- **Annual refresher training** — every calendar year
- **Phishing simulation** — at least 4 times per year (no advance notice)

Failure to complete training within 14 days of due date triggers a managed escalation. Repeated failures result in temporary suspension of system access until training is completed.

### 6.2 Reporting suspected incidents

If an employee suspects a security incident — phishing email, lost device, unusual account behavior, suspicious file — they must report it **within 1 hour** to:

- **Email**: security-incident@acme.example (monitored 24/7)
- **Slack**: #security-incidents channel
- **Phone (urgent)**: +49 800 555 0150 (24/7 hotline)

There is no penalty for reporting a false alarm. There is significant penalty for failing to report a real incident.

### 6.3 Lost or stolen devices

Lost or stolen devices must be reported within **30 minutes** of discovery. ISO will:

1. Trigger remote wipe within 5 minutes of report
2. Revoke all active sessions associated with the device
3. Issue a replacement device within 1 business day
4. Open a forensics review if Confidential or Restricted data was on the device

Employees are not financially liable for lost or stolen devices unless gross negligence is established.

---

## 7. Software and tooling

### 7.1 Approved software

A list of pre-approved software is maintained at https://acme.example/it/software. Installing software outside this list requires:

- Submission of a software request ticket
- Review by IT and ISO (typical turnaround: 5 business days)
- Approval by direct manager

Browser extensions are subject to the same approval process. Approval list is reviewed quarterly; previously-approved extensions can be revoked if a new vulnerability is discovered.

### 7.2 AI tools

Use of AI tools (ChatGPT, Claude, Copilot, etc.) is governed by the AI Acceptable Use addendum (separate document). In summary:

- Public LLM web interfaces (chatgpt.com, claude.ai) must not receive Confidential or Restricted data
- Approved enterprise endpoints (corporate-licensed Copilot, internal RAG tools) may be used per their specific data classification
- Any AI-generated content used in customer-facing deliverables must be reviewed and edited by a human author

---

## 8. Policy exceptions

Exceptions to this policy are granted only with written ISO approval and a documented compensating control. Exceptions are time-bound (maximum 90 days) and reviewed at expiration. There is no perpetual exception.

---

## 9. Contacts

| Topic | Channel |
| --- | --- |
| Security incidents (urgent) | +49 800 555 0150 |
| Security incidents (email) | security-incident@acme.example |
| General IT helpdesk | it-helpdesk@acme.example |
| ISO (policy questions) | iso@acme.example |
| Anonymous reporting | https://acme.example/ethics |

---

*This policy is reviewed annually. Mid-cycle updates are published with version number increment and announcement on the company-all channel.*
