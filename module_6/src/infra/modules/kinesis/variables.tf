variable "stream_name" {
  type        = string
  description = "Kinesis stream name"
}

variable "shard_count" {
  type        = number
  description = "Kinesis stream shard count"
}

variable "retention_period" {
  type        = number
  description = "Kinesis stream retention period"
}

variable "shard_level_metrics" {
    type        = list(string)
    description = "Kinesis stream shard level metrics"
    default = [
    "IncomingBytes",
    "OutgoingBytes",
    "OutgoingRecords",
    "ReadProvisionedThroughputExceeded",
    "WriteProvisionedThroughputExceeded",
    "IncomingRecords",
    "IteratorAgeMilliseconds"
  ]
}

variable "tags" {
  description = "Tags for kinesis stream"
  default     = "mlops-zoomcamp"
}

// you will have to provide the actual value when you call the variable...which seems a little silly but ok
