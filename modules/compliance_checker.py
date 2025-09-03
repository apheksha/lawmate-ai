def check_compliance(text, regulation="GDPR"):
    findings = []
    if "personal data" in text.lower() and regulation == "GDPR":
        findings.append("May violate GDPR data protection rules.")
    if "cross-border transfer" in text.lower() and regulation == "GDPR":
        findings.append("Cross-border transfer may be non-compliant.")
    if "employee data" in text.lower() and regulation == "IT Act":
        findings.append("May violate IT Act 2000 employee data provisions.")
    return findings
