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

class EnvironmentEvolverRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type EnvironmentEvolverRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type EnvironmentEvolverRequest
    let len;
    let data = new EnvironmentEvolverRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/EnvironmentEvolverRequest';
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
    const resolved = new EnvironmentEvolverRequest(null);
    return resolved;
    }
};

class EnvironmentEvolverResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.finished = null;
    }
    else {
      if (initObj.hasOwnProperty('finished')) {
        this.finished = initObj.finished
      }
      else {
        this.finished = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type EnvironmentEvolverResponse
    // Serialize message field [finished]
    bufferOffset = _serializer.bool(obj.finished, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type EnvironmentEvolverResponse
    let len;
    let data = new EnvironmentEvolverResponse(null);
    // Deserialize message field [finished]
    data.finished = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/EnvironmentEvolverResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e2797c797e620a950ba704492d72873b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool finished
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new EnvironmentEvolverResponse(null);
    if (msg.finished !== undefined) {
      resolved.finished = msg.finished;
    }
    else {
      resolved.finished = false
    }

    return resolved;
    }
};

module.exports = {
  Request: EnvironmentEvolverRequest,
  Response: EnvironmentEvolverResponse,
  md5sum() { return 'e2797c797e620a950ba704492d72873b'; },
  datatype() { return 'gazebo_radiation_plugins/EnvironmentEvolver'; }
};
