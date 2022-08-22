use rusoto_core::region::Region;
use rusoto_core::ByteStream;
use rusoto_s3::{
    AbortMultipartUploadRequest, CompleteMultipartUploadRequest, CompletedMultipartUpload,
    CompletedPart, CreateMultipartUploadRequest, UploadPartRequest,
};
use rusoto_s3::{S3Client, S3};
use std::env;
use std::io::{Error, Write};
use std::time::Duration;

use std::io::prelude::*;

pub struct S3File {
    bucket_name: String,
    object_key: String,
    s3_client: S3Client,
    upload_id: String,
    completed_parts: Vec<CompletedPart>,
    part_number: i64,
    buff: Vec<u8>,
    completed: bool,
    part_size: usize,
}

impl Drop for S3File {
    fn drop(&mut self) {
        unsafe {
            self.complete();
        }
    }
}

impl S3File {
    pub fn create(filename: String) -> S3File {
        let region = match env::var("S3_ENDPOINT_URL") {
            Ok(endpoint) => Region::Custom {
                name: "custom".to_string(),
                endpoint,
            },
            Err(e) => Region::default(),
        };

        let path: Vec<&str> = filename.strip_prefix("s3://").unwrap().split("/").collect();
        let bucket_name: String = path[0].to_string();
        let object_key: String = path[1..].join("/");

        let s3_client = S3Client::new(region);

        let part_size = 10 * 1024 * 1024; // note that max part count is 10k so we can upload up to 100_000 MiB
        let timeout = Duration::from_secs(10);
        let data_timeout = Duration::from_secs(300);

        let mut completed_parts: Vec<CompletedPart> = Vec::new();
        let upload_id = &s3_client
            .create_multipart_upload(CreateMultipartUploadRequest {
                bucket: bucket_name.clone(),
                key: object_key.clone(),
                //content_type: Some(meta.content_type),
                //content_disposition: meta.content_disposition,
                //content_language: meta.content_language,
                ..Default::default()
            })
            .with_timeout(timeout)
            .sync()
            .unwrap()
            .upload_id
            .expect("no upload ID");

        let buff = Vec::new();

        S3File {
            bucket_name,
            object_key,
            s3_client,
            upload_id: upload_id.to_string(),
            completed_parts,
            part_number: 0,
            buff,
            completed: false,
            part_size,
        }
    }

    fn write_buff(&mut self) {
        if self.buff.len() == 0 {
            return;
        }

        let mut buff = self.buff.to_owned();
        let data_timeout = Duration::from_secs(300);

        let result = self
            .s3_client
            .upload_part(UploadPartRequest {
                body: Some(ByteStream::from(buff)),
                bucket: self.bucket_name.clone(),
                key: self.object_key.clone(),
                part_number: self.part_number as i64,
                upload_id: self.upload_id.clone(),
                ..Default::default()
            })
            .with_timeout(data_timeout)
            .sync()
            .unwrap();

        self.completed_parts.push(CompletedPart {
            e_tag: result.e_tag,
            part_number: Some(self.part_number as i64),
        });

        self.part_number += 1;
        self.buff = Vec::new();
    }

    pub fn complete(&mut self) {
        if !self.completed {
            self.write_buff();
            let timeout = Duration::from_secs(10);
            self.s3_client
                .complete_multipart_upload(CompleteMultipartUploadRequest {
                    bucket: self.bucket_name.clone(),
                    key: self.object_key.clone(),
                    upload_id: self.upload_id.clone(),
                    multipart_upload: Some(CompletedMultipartUpload {
                        parts: Some(self.completed_parts.clone()),
                    }),
                    ..Default::default()
                })
                .with_timeout(timeout)
                .sync()
                .unwrap();
            self.completed = true;
        }
    }
}

impl Write for S3File {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Error> {
        self.buff.extend_from_slice(buf);

        if self.buff.len() > self.part_size {
            self.write_buff();
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), Error> {
        //self.write_buff();
        Ok(())
    }
}
