// Auto-generated. Do not edit!

// (in-package gazebo_radiation_plugins.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class GenYamlsFromWorldRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.folder = null;
      this.world_filename = null;
      this.model_subset_filename = null;
      this.default_bend = null;
      this.default_rust = null;
      this.default_radiation = null;
    }
    else {
      if (initObj.hasOwnProperty('folder')) {
        this.folder = initObj.folder
      }
      else {
        this.folder = '';
      }
      if (initObj.hasOwnProperty('world_filename')) {
        this.world_filename = initObj.world_filename
      }
      else {
        this.world_filename = '';
      }
      if (initObj.hasOwnProperty('model_subset_filename')) {
        this.model_subset_filename = initObj.model_subset_filename
      }
      else {
        this.model_subset_filename = '';
      }
      if (initObj.hasOwnProperty('default_bend')) {
        this.default_bend = initObj.default_bend
      }
      else {
        this.default_bend = 0.0;
      }
      if (initObj.hasOwnProperty('default_rust')) {
        this.default_rust = initObj.default_rust
      }
      else {
        this.default_rust = 0.0;
      }
      if (initObj.hasOwnProperty('default_radiation')) {
        this.default_radiation = initObj.default_radiation
      }
      else {
        this.default_radiation = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GenYamlsFromWorldRequest
    // Serialize message field [folder]
    bufferOffset = _serializer.string(obj.folder, buffer, bufferOffset);
    // Serialize message field [world_filename]
    bufferOffset = _serializer.string(obj.world_filename, buffer, bufferOffset);
    // Serialize message field [model_subset_filename]
    bufferOffset = _serializer.string(obj.model_subset_filename, buffer, bufferOffset);
    // Serialize message field [default_bend]
    bufferOffset = _serializer.float32(obj.default_bend, buffer, bufferOffset);
    // Serialize message field [default_rust]
    bufferOffset = _serializer.float32(obj.default_rust, buffer, bufferOffset);
    // Serialize message field [default_radiation]
    bufferOffset = _serializer.float32(obj.default_radiation, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GenYamlsFromWorldRequest
    let len;
    let data = new GenYamlsFromWorldRequest(null);
    // Deserialize message field [folder]
    data.folder = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [world_filename]
    data.world_filename = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [model_subset_filename]
    data.model_subset_filename = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [default_bend]
    data.default_bend = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [default_rust]
    data.default_rust = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [default_radiation]
    data.default_radiation = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.folder);
    length += _getByteLength(object.world_filename);
    length += _getByteLength(object.model_subset_filename);
    return length + 24;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/GenYamlsFromWorldRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '96e84490834f1029806f056f6761f002';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string folder
    string world_filename
    string model_subset_filename
    float32 default_bend
    float32 default_rust
    float32 default_radiation
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GenYamlsFromWorldRequest(null);
    if (msg.folder !== undefined) {
      resolved.folder = msg.folder;
    }
    else {
      resolved.folder = ''
    }

    if (msg.world_filename !== undefined) {
      resolved.world_filename = msg.world_filename;
    }
    else {
      resolved.world_filename = ''
    }

    if (msg.model_subset_filename !== undefined) {
      resolved.model_subset_filename = msg.model_subset_filename;
    }
    else {
      resolved.model_subset_filename = ''
    }

    if (msg.default_bend !== undefined) {
      resolved.default_bend = msg.default_bend;
    }
    else {
      resolved.default_bend = 0.0
    }

    if (msg.default_rust !== undefined) {
      resolved.default_rust = msg.default_rust;
    }
    else {
      resolved.default_rust = 0.0
    }

    if (msg.default_radiation !== undefined) {
      resolved.default_radiation = msg.default_radiation;
    }
    else {
      resolved.default_radiation = 0.0
    }

    return resolved;
    }
};

class GenYamlsFromWorldResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GenYamlsFromWorldResponse
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GenYamlsFromWorldResponse
    let len;
    let data = new GenYamlsFromWorldResponse(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/GenYamlsFromWorldResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GenYamlsFromWorldResponse(null);
    return resolved;
    }
};

module.exports = {
  Request: GenYamlsFromWorldRequest,
  Response: GenYamlsFromWorldResponse,
  md5sum() { return '96e84490834f1029806f056f6761f002'; },
  datatype() { return 'gazebo_radiation_plugins/GenYamlsFromWorld'; }
};
