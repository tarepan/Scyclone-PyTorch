const path = require('path');
const fs = require('fs');
const core = require('@actions/core');
const AWS = require('aws-sdk');

async function run() {
    try {
        // Get inputs
        const pathJobDefInput = core.getInput("path-task-definition", {
            required: true,
        });
        const adressContainer = core.getInput("adress-container", { required: true });

        // Parse the task definition
        const root = process.env.GITHUB_WORKSPACE ? process.env.GITHUB_WORKSPACE : ""; // for Node12
        const jobDefPath = path.isAbsolute(pathJobDefInput)
            ? pathJobDefInput
            : path.join(root, pathJobDefInput);
        if (!fs.existsSync(jobDefPath)) {
            throw new Error(`Task definition file does not exist: ${pathJobDefInput}`);
        }
        const baseJobDef = JSON.parse(
            fs.readFileSync(jobDefPath, "utf-8")
        );

        // Template Docker image adress
        baseJobDef.containerProperties.image = adressContainer

        // Register jobDef to Batch
        const batch = new AWS.Batch();
        await batch.registerJobDefinition(baseJobDef).promise();
    } catch (error) {
        core.setFailed(error.message);
    }
}

module.exports = run;

/* istanbul ignore next */
if (require.main === module) {
    run();
}