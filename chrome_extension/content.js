/**
 * PhishGuard Chrome Extension - Content Script
 * =============================================
 *
 * This content script runs on all pages and provides additional
 * detection capabilities based on page content.
 *
 * Features:
 * - Detect login forms (potential credential harvesting)
 * - Monitor form submissions
 * - Detect password fields on suspicious pages
 */

(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        // Track forms on the page
        trackForms: true,
        // Alert on suspicious form submissions
        alertOnSubmit: true,
        // Check for password fields
        detectPasswordFields: true
    };

    // State
    let formsTracked = [];
    let hasPasswordField = false;

    /**
     * Initialize content script
     */
    function init() {
        console.log('PhishGuard: Content script initialized on', window.location.href);

        // Check for password fields
        checkForPasswordFields();

        // Track forms if enabled
        if (CONFIG.trackForms) {
            trackForms();
        }

        // Listen for form submissions
        if (CONFIG.alertOnSubmit) {
            monitorFormSubmissions();
        }

        // Report page info to background
        reportPageInfo();
    }

    /**
     * Check for password input fields
     */
    function checkForPasswordFields() {
        const passwordFields = document.querySelectorAll('input[type="password"]');
        hasPasswordField = passwordFields.length > 0;

        if (hasPasswordField) {
            console.log('PhishGuard: Password field detected on page');
            // You could enhance this by checking if the page looks suspicious
        }
    }

    /**
     * Track forms on the page
     */
    function trackForms() {
        const forms = document.querySelectorAll('form');
        formsTracked = Array.from(forms).map(form => ({
            action: form.action || window.location.href,
            method: form.method || 'get',
            inputs: form.querySelectorAll('input').length,
            hasPassword: form.querySelector('input[type="password"]') !== null
        }));

        console.log('PhishGuard: Tracked', formsTracked.length, 'forms');
    }

    /**
     * Monitor form submissions
     */
    function monitorFormSubmissions() {
        document.addEventListener('submit', (event) => {
            const form = event.target;
            const formData = new FormData(form);

            console.log('PhishGuard: Form submitted', {
                action: form.action,
                hasPassword: form.querySelector('input[type="password"]') !== null
            });

            // Log submission for debugging
            // In production, you might want to notify the background script

            // Check if this looks like a credential form submission
            const hasCredentials = Array.from(formData.keys()).some(key =>
                ['password', 'pwd', 'pass', 'credential', 'login', 'email', 'username'].some(
                    cred => key.toLowerCase().includes(cred)
                )
            );

            if (hasCredentials) {
                console.log('PhishGuard: Credential form detected - notifying background');
                chrome.runtime.sendMessage({
                    type: 'FORM_SUBMITTED',
                    url: window.location.href,
                    formAction: form.action,
                    timestamp: Date.now()
                }).catch(() => {
                    // Background might not be available
                });
            }
        }, true);
    }

    /**
     * Report page information to background script
     */
    function reportPageInfo() {
        // This is a simple check - more sophisticated analysis could be added
        const pageInfo = {
            url: window.location.href,
            domain: window.location.hostname,
            title: document.title,
            hasPasswordField: hasPasswordField,
            formCount: formsTracked.length,
            timestamp: Date.now()
        };

        // Store in sessionStorage for popup access
        try {
            sessionStorage.setItem('phishguard_page_info', JSON.stringify(pageInfo));
        } catch (e) {
            // Storage might be full
        }
    }

    /**
     * Detect if the page appears to be impersonating another site
     */
    function detectImpersonation() {
        // Get page title
        const title = document.title.toLowerCase();

        // Check for impersonation indicators
        const impersonation_keywords = [
            'facebook login', 'apple account', 'microsoft account',
            'paypal login', 'amazon sign in', 'google account'
        ];

        const isImpersonating = impersonation_keywords.some(keyword =>
            title.includes(keyword) &&
            window.location.hostname.toLowerCase().includes(keyword.split(' ')[0]) === false
        );

        if (isImpersonating) {
            console.warn('PhishGuard: Possible brand impersonation detected!');

            // Notify background
            chrome.runtime.sendMessage({
                type: 'IMPERSINATION_DETECTED',
                url: window.location.href,
                title: document.title
            }).catch(() => {});

            return true;
        }

        return false;
    }

    /**
     * Check meta tags for security indicators
     */
    function checkMetaTags() {
        const securityMeta = {
            hasFrameBreaker: false,
            hasContentSecurity: false,
            isSecureFrame: false
        };

        // Check for frame-breaking scripts (often used in phishing)
        if (window.top !== window.self) {
            securityMeta.isSecureFrame = true;
        }

        // Check meta tags
        const metaTags = document.querySelectorAll('meta');

        for (const meta of metaTags) {
            const httpEquiv = meta.getAttribute('http-equiv') || '';
            if (httpEquiv.toLowerCase().includes('x-frame')) {
                securityMeta.hasFrameBreaker = true;
            }
        }

        return securityMeta;
    }

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Also run on dynamic content changes (for SPAs)
    const observer = new MutationObserver((mutations) => {
        let shouldRecheck = false;

        for (const mutation of mutations) {
            if (mutation.addedNodes.length > 0) {
                for (const node of mutation.addedNodes) {
                    if (node.nodeName === 'INPUT' ||
                        (node.querySelector && node.querySelector('input[type="password"]'))) {
                        shouldRecheck = true;
                        break;
                    }
                }
            }
        }

        if (shouldRecheck) {
            checkForPasswordFields();
            trackForms();
        }
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    console.log('PhishGuard: Content script loaded');
})();