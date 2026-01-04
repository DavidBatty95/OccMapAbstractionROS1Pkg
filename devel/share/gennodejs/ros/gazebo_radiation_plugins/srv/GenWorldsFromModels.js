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

class GenWorldsFromModelsRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.folder = null;
      this.model_filename = null;
    }
    else {
      if (initObj.hasOwnProperty('folder')) {
        this.folder = initObj.folder
      }
      else {
        this.folder = '';
      }
      if (initObj.hasOwnProperty('model_filename')) {
        this.model_filename = initObj.model_filename
      }
      else {
        this.model_filename = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GenWorldsFromModelsRequest
    // Serialize message field [folder]
    bufferOffset = _serializer.string(obj.folder, buffer, bufferOffset);
    // Serialize message field [model_filename]
    bufferOffset = _serializer.string(obj.model_filename, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GenWorldsFromModelsRequest
    let len;
    let data = new GenWorldsFromModelsRequest(null);
    // Deserialize message field [folder]
    data.folder = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [model_filename]
    data.model_filename = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.folder);
    length += _getByteLength(object.model_filename);
    return length + 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/GenWorldsFromModelsRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'aab9442741bfdbb3be9f5e426f1408ca';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string folder
    string model_filename
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GenWorldsFromModelsRequest(null);
    if (msg.folder !== undefined) {
      resolved.folder = msg.folder;
    }
    else {
      resolved.folder = ''
    }

    if (msg.model_filename !== undefined) {
      resolved.model_filename = msg.model_filename;
    }
    else {
      resolved.model_filename = ''
    }

    return resolved;
    }
};

class GenWorldsFromModelsResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GenWorldsFromModelsResponse
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GenWorldsFromModelsResponse
    let len;
    let data = new GenWorldsFromModelsResponse(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/GenWorldsFromModelsResponse';
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
    const resolved = new GenWorldsFromModelsResponse(null);
    return resolved;
    }
};

module.exports = {
  Request: GenWorldsFromModelsRequest,
  Response: GenWorldsFromModelsResponse,
  md5sum() { return 'aab9442741bfdbb3be9f5e426f1408ca'; },
  datatype() { return 'gazebo_radiation_plugins/GenWorldsFromModels'; }
};
