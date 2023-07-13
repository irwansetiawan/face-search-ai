
import { RekognitionClient, CompareFacesCommand } from "@aws-sdk/client-rekognition";
import { Request, Response } from "express";
import * as fs from 'fs';

const client = new RekognitionClient({
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
    }
});

export async function compareFace(req: Request, res: Response) {
    const files = req.files as { [fieldname: string]: Express.Multer.File[] };
    console.log(files);
    const input = {
        SourceImage: {
          Bytes: fs.readFileSync(files.source[0].path),
        },
        TargetImage: {
          Bytes: fs.readFileSync(files.target[0].path),
        },
        SimilarityThreshold: 75,
        QualityFilter: "NONE",
    };
    const command = new CompareFacesCommand(input);
    let response;
    try {
        response = await client.send(command);
        res.status(200).json(response);
    } catch(e) {
        res.status(400).json(e);
    }
    
};
