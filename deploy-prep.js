const path = require('path');
const fs = require('fs').promises;

async function incrementVersion() {
    try {
        const versionPath = path.join(__dirname, 'version.json');
        const versionData = await fs.readFile(versionPath, 'utf8');
        const versionObj = JSON.parse(versionData);

        // Increment patch version (e.g., 1.0.0 -> 1.0.1)
        const [major, minor, patch] = versionObj.version.split('.').map(Number);
        const oldVersion = versionObj.version;
        const newVersion = `${major}.${minor}.${patch + 1}`;

        versionObj.version = newVersion;
        await fs.writeFile(versionPath, JSON.stringify(versionObj, null, 2), 'utf8');

        console.log(`Version incremented from ${oldVersion} to ${newVersion}`);
        return newVersion;
    } catch (error) {
        console.error('Error incrementing version:', error.message);
        return '1.0.0'; // fallback
    }
}

async function updateCacheBusting(htmlPath, version) {
    try {
        const htmlContent = await fs.readFile(htmlPath, 'utf8');
        const timestamp = Date.now();

        // Replace version parameters with timestamp for cache busting
        let updatedContent = htmlContent.replace(/\?v=[^"'\s)]+/g, `?v=${timestamp}`);

        // Update version display in HTML
        updatedContent = updatedContent.replace(/v\d+\.\d+\.\d+/, `v${version}`);

        await fs.writeFile(htmlPath, updatedContent, 'utf8');
        console.log(`Updated cache-busting parameters with timestamp: ${timestamp}`);
        console.log(`Updated version display to: v${version}`);
    } catch (error) {
        console.error('Error updating cache busting:', error.message);
    }
}

async function prepareDeploy() {
    try {
        // Increment version and update cache-busting parameters in HTML before deploying
        const htmlPath = path.join(__dirname, 'dogbarkingdetector', 'index.html');
        const newVersion = await incrementVersion();
        await updateCacheBusting(htmlPath, newVersion);

        console.log('Deploy preparation complete.');
    } catch (error) {
        console.error('Deploy preparation failed:', error.message);
        process.exit(1);
    }
}

prepareDeploy();